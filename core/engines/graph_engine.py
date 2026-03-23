# core/engines/graph_engine.py
"""Graph analytics engine for detecting fraud rings and network-based anomalies."""
import sqlite3
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

import networkx as nx
from community import community_louvain  # python-louvain package

from core.engines.base import BaseDetectionEngine
from core.models.beneficiary import FraudIndicator


class GraphEngine(BaseDetectionEngine):
    """
    Graph-based fraud detection using NetworkX.
    Detects: Fraud rings, money mules, collusion networks, agent-beneficiary clusters.
    """
    
    def __init__(self, db_path: str = "data/processed/fraud_system.db", weight: float = 0.20):
        super().__init__(name="graph", weight=weight, db_path=db_path)
        
        # Graph construction parameters
        self.min_ring_size = 3
        self.max_ring_size = 50  # Ignore massive components (likely legitimate groups)
        
        # Suspicious patterns
        self.mule_threshold = 0.8  # Betweenness centrality threshold
        self.collusion_edge_threshold = 3  # Min edges between beneficiaries for collusion
        
        self.graphs = {}  # Store different graph types
        self.communities = {}
        self.centrality_scores = {}
        
    def train(self, data: Optional[Any] = None):
        """Build graphs from database relationships."""
        self._build_all_graphs()
        self._detect_communities()
        self._calculate_centrality()
        self.is_trained = True
        
    def _build_all_graphs(self):
        """Construct multiple graph views from relational data."""
        print("Building fraud detection graphs...")
        
        self.graphs = {
            'shared_address': self._build_address_graph(),
            'shared_bank': self._build_bank_graph(),
            'shared_phone': self._build_phone_graph(),
            'agent_beneficiary': self._build_agent_graph(),
            'transaction_pattern': self._build_transaction_graph()
        }
        
    def _build_address_graph(self) -> nx.Graph:
        """Graph where edges exist if beneficiaries share same address."""
        G = nx.Graph()
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            # Get address clusters
            cursor.execute("""
                SELECT address, GROUP_CONCAT(beneficiary_id, ',') as ids
                FROM beneficiaries
                GROUP BY address
                HAVING COUNT(*) >= 2
            """)
            
            for row in cursor.fetchall():
                ids = row['ids'].split(',')
                # Create clique (fully connected subgraaph) for this address
                for i in range(len(ids)):
                    for j in range(i+1, len(ids)):
                        G.add_edge(ids[i], ids[j], 
                                  relation='same_address', 
                                  weight=1.0)
        
        return G
    
    def _build_bank_graph(self) -> nx.Graph:
        """Graph for shared bank accounts (strong fraud signal)."""
        G = nx.Graph()
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT bank_hash, GROUP_CONCAT(beneficiary_id, ',') as ids, COUNT(*) as cnt
                FROM beneficiaries
                GROUP BY bank_hash
                HAVING cnt >= 2
            """)
            
            for row in cursor.fetchall():
                ids = row['ids'].split(',')
                edge_weight = min(1.0, row['cnt'] / 5)  # Higher count = stronger edge
                
                for i in range(len(ids)):
                    for j in range(i+1, len(ids)):
                        G.add_edge(ids[i], ids[j],
                                  relation='shared_account',
                                  weight=edge_weight)
        
        return G
    
    def _build_phone_graph(self) -> nx.Graph:
        """Graph for shared phone numbers."""
        G = nx.Graph()
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT phone_hash, GROUP_CONCAT(beneficiary_id, ',') as ids
                FROM beneficiaries
                WHERE phone_hash IS NOT NULL
                GROUP BY phone_hash
                HAVING COUNT(*) >= 2
            """)
            
            for row in cursor.fetchall():
                ids = row['ids'].split(',')
                for i in range(len(ids)):
                    for j in range(i+1, len(ids)):
                        G.add_edge(ids[i], ids[j],
                                  relation='shared_phone',
                                  weight=0.7)
        
        return G
    
    def _build_agent_graph(self) -> nx.Graph:
        """
        Bipartite graph: Beneficiaries <-> Agents.
        Detects suspicious agent-beneficiary clusters.
        """
        G = nx.Graph()  # Bipartite graphs use node attributes, not separate class
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get all agent-beneficiary relationships
            cursor.execute("""
                SELECT DISTINCT beneficiary_id, agent_id, COUNT(*) as txn_count
                FROM transactions
                GROUP BY beneficiary_id, agent_id
            """)
            
            for row in cursor.fetchall():
                ben_id = f"B_{row['beneficiary_id']}"
                agt_id = f"A_{row['agent_id']}"
                
                G.add_edge(ben_id, agt_id, 
                          weight=row['txn_count'],
                          relation='agent_service')
        
        return G
    
    def _build_transaction_graph(self) -> nx.DiGraph:
        """
        Directed graph of transaction flows.
        Detects money laundering patterns.
        """
        G = nx.DiGraph()
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Beneficiaries who transact at same agents close in time (possible collusion)
            cursor.execute("""
                SELECT t1.beneficiary_id as b1, t2.beneficiary_id as b2,
                       t1.agent_id, t1.transaction_date
                FROM transactions t1
                JOIN transactions t2 ON t1.agent_id = t2.agent_id
                    AND t1.beneficiary_id < t2.beneficiary_id
                    AND ABS(JULIANDAY(t1.transaction_date) - JULIANDAY(t2.transaction_date)) < 1
                WHERE t1.status = 'success' AND t2.status = 'success'
                LIMIT 10000  -- Sampling for performance
            """)
            
            for row in cursor.fetchall():
                G.add_edge(row['b1'], row['b2'],
                          agent=row['agent_id'],
                          time=row['transaction_date'])
        
        return G
    
    def _detect_communities(self):
        """Detect fraud rings using Louvain community detection."""
        print("Detecting communities (fraud rings)...")
        
        # Use combined graph for community detection
        combined = nx.Graph()
        for G in self.graphs.values():
            if isinstance(G, nx.Graph) and not G.is_directed():
                combined = nx.compose(combined, G)
        
        if len(combined.edges) > 0:
            # Louvain method for community detection
            partition = community_louvain.best_partition(combined)
            self.communities = partition
            
            # Group by community
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            
            # Score communities by density and size
            self.community_scores = {}
            for comm_id, nodes in communities.items():
                if len(nodes) >= self.min_ring_size:
                    subgraph = combined.subgraph(nodes)
                    density = nx.density(subgraph)
                    self.community_scores[comm_id] = {
                        'size': len(nodes),
                        'density': density,
                        'nodes': nodes
                    }
        else:
            self.communities = {}
            self.community_scores = {}
    
    def _calculate_centrality(self):
        """Calculate network centrality - OPTIMIZED with sampling."""
        from tqdm import tqdm
        print("Calculating centrality metrics (this may take a minute)...")
    
        # Use only undirected graphs for centrality
        combined = nx.Graph()
        for name, G in self.graphs.items():
            if isinstance(G, nx.Graph) and not G.is_directed():
                combined = nx.compose(combined, G)
    
        print(f"  Graph size: {len(combined.nodes):,} nodes, {len(combined.edges):,} edges")
    
        if len(combined.nodes) > 0:
            # For large graphs, use sampling (betweenness on full graph is too slow)
            if len(combined.nodes) > 1000:
                print("  Large graph detected - using sampling for performance...")
                # Sample 1000 nodes with highest degree
                degrees = dict(combined.degree())
                top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:1000]
                subgraph = combined.subgraph(top_nodes)
                print(f"  Analyzing subgraph of {len(subgraph.nodes):,} high-degree nodes...")
                self.centrality_scores = nx.betweenness_centrality(subgraph)
            
                # Assign 0 to nodes not in sample
                for node in combined.nodes:
                    if node not in self.centrality_scores:
                        self.centrality_scores[node] = 0.0
            else:
                self.centrality_scores = nx.betweenness_centrality(combined)
            
            print(f"  ✓ Calculated centrality for {len(self.centrality_scores):,} nodes")
        else:
            self.centrality_scores = {}
    
    def analyze(self, beneficiary_id: str) -> FraudIndicator:
        """Analyze beneficiary's position in fraud networks."""
        violations = []
        score = 0
        
        # Check 1: Membership in suspicious community (fraud ring)
        comm_check = self._check_community_membership(beneficiary_id)
        if comm_check:
            violations.append(comm_check)
            score = max(score, comm_check['score_contribution'])
        
        # Check 2: High centrality (potential mule)
        mule_check = self._check_mule_indicator(beneficiary_id)
        if mule_check:
            violations.append(mule_check)
            score = max(score, mule_check['score_contribution'])
        
        # Check 3: Direct connections to known fraudsters
        connection_check = self._check_fraud_connections(beneficiary_id)
        if connection_check:
            violations.append(connection_check)
            score = max(score, connection_check['score_contribution'])
        
        # Check 4: Suspicious agent clustering
        agent_check = self._check_agent_collusion(beneficiary_id)
        if agent_check:
            violations.append(agent_check)
            score = max(score, agent_check['score_contribution'])
        
        # Check 5: Multi-hop relations (2nd degree fraud connections)
        hop_check = self._check_multi_hop_fraud(beneficiary_id)
        if hop_check:
            violations.append(hop_check)
            score = max(score, hop_check['score_contribution'])
        
        if violations:
            primary = max(violations, key=lambda x: x['score_contribution'])
            description = f"Network anomalies ({len(violations)}): {primary['reason']}"
        else:
            description = "No network anomalies detected"
        
        return FraudIndicator(
            engine=self.name,
            score=min(100, score),
            severity=self._score_to_severity(score),
            description=description,
            details={
                'violation_count': len(violations),
                'violations': violations,
                'centrality_score': self.centrality_scores.get(beneficiary_id, 0),
                'connections': self._get_connection_count(beneficiary_id)
            }
        )
    
    def analyze_batch(self, beneficiary_ids: List[str]) -> List[FraudIndicator]:
        """Batch analysis (graphs already built)."""
        return [self.analyze(bid) for bid in beneficiary_ids]
    
    def _check_community_membership(self, beneficiary_id: str) -> Optional[Dict]:
        """Check if beneficiary is in a suspicious fraud ring."""
        if beneficiary_id not in self.communities:
            return None
        
        comm_id = self.communities[beneficiary_id]
        if comm_id not in self.community_scores:
            return None
        
        comm_info = self.community_scores[comm_id]
        size = comm_info['size']
        density = comm_info['density']
        
        # Suspicious if: small tight-knit group OR large dense group
        if size >= self.min_ring_size:
            if size <= 10 and density > 0.5:  # Tight fraud ring
                return {
                    'type': 'fraud_ring_tight',
                    'reason': f"Member of tight-knit group: {size} people, {density:.0%} connected",
                    'score_contribution': min(95, 70 + size*2),
                    'community_size': size,
                    'density': density,
                    'feature': 'network_community',
                    'evidence': f"Part of {size}-person fraud ring with high internal connectivity"
                }
            elif size > 20 and density > 0.3:  # Large organized ring
                return {
                    'type': 'fraud_ring_large',
                    'reason': f"Member of large organized network: {size} people",
                    'score_contribution': 85,
                    'community_size': size,
                    'density': density,
                    'feature': 'network_community',
                    'evidence': "Connected to large coordinated fraud network"
                }
        return None
    
    def _check_mule_indicator(self, beneficiary_id: str) -> Optional[Dict]:
        """Check for money mule characteristics (high betweenness)."""
        if beneficiary_id not in self.centrality_scores:
            return None
        
        centrality = self.centrality_scores[beneficiary_id]
        
        if centrality > self.mule_threshold:
            return {
                'type': 'network_mule',
                'reason': f"Potential money mule (Betweenness: {centrality:.3f})",
                'score_contribution': min(90, 60 + int(centrality * 30)),
                'centrality': centrality,
                'feature': 'network_centrality',
                'evidence': f"Acts as bridge between {self._count_bridges(beneficiary_id)} fraud clusters"
            }
        elif centrality > 0.5:
            return {
                'type': 'network_connector',
                'reason': f"Suspicious network connectivity (Betweenness: {centrality:.3f})",
                'score_contribution': 65,
                'centrality': centrality,
                'feature': 'network_centrality',
                'evidence': "Unusually central position in transaction network"
            }
        return None
    
    def _check_fraud_connections(self, beneficiary_id: str) -> Optional[Dict]:
        """Check direct connections to known fraud patterns."""
        # Check shared_bank graph (strongest signal)
        if 'shared_bank' in self.graphs:
            G = self.graphs['shared_bank']
            if beneficiary_id in G:
                neighbors = list(G.neighbors(beneficiary_id))
                fraud_neighbors = [n for n in neighbors if self._is_known_fraud(n)]
                
                if fraud_neighbors:
                    return {
                        'type': 'fraud_network_neighbor',
                        'reason': f"Direct connection to {len(fraud_neighbors)} known fraudster(s)",
                        'score_contribution': min(100, 70 + len(fraud_neighbors) * 10),
                        'fraud_connections': len(fraud_neighbors),
                        'feature': 'network_proximity',
                        'evidence': f"Shares bank/account with flagged beneficiaries"
                    }
        return None
    
    def _check_agent_collusion(self, beneficiary_id: str) -> Optional[Dict]:
        """Check for suspicious agent-beneficiary patterns."""
        if 'agent_beneficiary' not in self.graphs:
            return None
        
        G = self.graphs['agent_beneficiary']
        ben_node = f"B_{beneficiary_id}"
        
        if ben_node not in G:
            return None
        
        agents = [n for n in G.neighbors(ben_node) if n.startswith('A_')]
        
        if len(agents) > 5:  # Uses many different agents
            return {
                'type': 'agent_hopping',
                'reason': f"Uses {len(agents)} different agents (unusual)",
                'score_contribution': min(75, 50 + len(agents) * 3),
                'agent_count': len(agents),
                'feature': 'agent_behavior',
                'evidence': "Frequent switching between service providers"
            }
        
        # Check if connected to high-risk agent
        high_risk_agents = [a for a in agents if self._is_high_risk_agent(a)]
        if high_risk_agents:
            return {
                'type': 'high_risk_agent',
                'reason': f"Connected to {len(high_risk_agents)} flagged agent(s)",
                'score_contribution': 70,
                'feature': 'agent_risk',
                'evidence': "Transacts through agents with high fraud scores"
            }
        
        return None
    
    def _check_multi_hop_fraud(self, beneficiary_id: str) -> Optional[Dict]:
        """Check 2nd degree connections to fraud (friend of friend)."""
        # Build ego network
        all_connections = set()
        
        for G in self.graphs.values():
            if beneficiary_id in G:
                # 1st degree
                first_degree = set(G.neighbors(beneficiary_id))
                all_connections.update(first_degree)
                
                # 2nd degree (limited)
                for neighbor in list(first_degree)[:10]:  # Sample for performance
                    if neighbor in G:
                        second_degree = set(G.neighbors(neighbor))
                        fraud_in_2nd = [n for n in second_degree if self._is_known_fraud(n)]
                        if fraud_in_2nd:
                            return {
                                'type': 'second_degree_fraud',
                                'reason': "2nd-degree connection to known fraudster",
                                'score_contribution': 50,
                                'feature': 'network_proximity',
                                'evidence': "Connected to fraudster through intermediaries"
                            }
        
        return None
    
    def _is_known_fraud(self, beneficiary_id: str) -> bool:
        """Check database if beneficiary is flagged."""
        # In real system, query fraud_results table
        # For now, check if in high-centrality or high-degree
        if beneficiary_id in self.centrality_scores:
            return self.centrality_scores[beneficiary_id] > 0.7
        return False
    
    def _is_high_risk_agent(self, agent_id: str) -> bool:
        """Check if agent has high fraud score."""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT fraud_score FROM agents WHERE agent_id = ?
            """, (agent_id.replace('A_', ''),))
            row = cursor.fetchone()
            if row and row[0] > 70:
                return True
        return False
    
    def _get_connection_count(self, beneficiary_id: str) -> int:
        """Count total unique connections across all graphs."""
        connections = set()
        for G in self.graphs.values():
            if beneficiary_id in G:
                connections.update(G.neighbors(beneficiary_id))
        return len(connections)
    
    def _count_bridges(self, beneficiary_id: str) -> int:
        """Count how many different fraud clusters this node connects."""
        if 'shared_bank' not in self.graphs:
            return 0
        
        G = self.graphs['shared_bank']
        if beneficiary_id not in G:
            return 0
        
        # Find neighbors in different communities
        communities = set()
        for neighbor in G.neighbors(beneficiary_id):
            if neighbor in self.communities:
                communities.add(self.communities[neighbor])
        
        return len(communities)
    
    def _score_to_severity(self, score: float) -> str:
        if score >= 80:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        return "low"
    
    def get_network_statistics(self) -> Dict:
        """Get overall network statistics."""
        stats = {}
        for name, G in self.graphs.items():
            stats[name] = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G) if G.number_of_nodes() > 1 else 0
            }
        
        stats['communities_detected'] = len(self.community_scores)
        stats['avg_centrality'] = sum(self.centrality_scores.values()) / len(self.centrality_scores) if self.centrality_scores else 0
        
        return stats


# Install dependency note
if __name__ == "__main__":
    print("Graph Engine Test")
    print("Note: Install python-louvain: pip install python-louvain networkx")
    
    engine = GraphEngine()
    engine.train()
    
    print(f"Graph Stats: {engine.get_network_statistics()}")
    
    # Test on random beneficiary
    import random
    with engine.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT beneficiary_id FROM beneficiaries LIMIT 100")
        ids = [r[0] for r in cursor.fetchall()]
        test_id = random.choice(ids)
    
    result = engine.analyze(test_id)
    print(f"\nTest: {test_id}")
    print(f"Score: {result.score}")
    print(f"Description: {result.description}")