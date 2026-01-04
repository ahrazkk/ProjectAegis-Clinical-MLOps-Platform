"""
Knowledge Graph Service for Drug-Drug Interaction Analysis
Handles Neo4j connection and Cypher queries for the DDI Knowledge Graph
"""

from neo4j import GraphDatabase
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """Service for interacting with the Neo4j Knowledge Graph"""
    
    _driver = None
    
    @classmethod
    def get_driver(cls):
        """Get or create Neo4j driver connection"""
        if cls._driver is None:
            try:
                neo4j_config = getattr(settings, 'NEO4J_CONFIG', {})
                uri = neo4j_config.get('uri', 'bolt://localhost:7687')
                user = neo4j_config.get('user', 'neo4j')
                password = neo4j_config.get('password', 'password')
                
                cls._driver = GraphDatabase.driver(uri, auth=(user, password))
                # Verify connection
                cls._driver.verify_connectivity()
                logger.info("Successfully connected to Neo4j")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                cls._driver = None
        return cls._driver
    
    @classmethod
    def close(cls):
        """Close the Neo4j driver"""
        if cls._driver:
            cls._driver.close()
            cls._driver = None
    
    @classmethod
    def is_connected(cls):
        """Check if Neo4j is connected"""
        try:
            driver = cls.get_driver()
            if driver:
                driver.verify_connectivity()
                return True
        except:
            pass
        return False
    
    @classmethod
    def run_query(cls, query: str, parameters: dict = None):
        """Run a Cypher query and return results"""
        driver = cls.get_driver()
        if not driver:
            return []
        
        try:
            with driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    # ========== Schema Creation ==========
    
    @classmethod
    def create_schema(cls):
        """Create the DDI Knowledge Graph schema with constraints and indexes"""
        queries = [
            # Constraints for unique identifiers
            "CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (d:Drug) REQUIRE d.drugbank_id IS UNIQUE",
            "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT target_id IF NOT EXISTS FOR (t:Target) REQUIRE t.uniprot_id IS UNIQUE",
            "CREATE CONSTRAINT enzyme_id IF NOT EXISTS FOR (e:Enzyme) REQUIRE e.uniprot_id IS UNIQUE",
            "CREATE CONSTRAINT side_effect_id IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.umls_id IS UNIQUE",
            
            # Indexes for faster lookups
            "CREATE INDEX drug_smiles IF NOT EXISTS FOR (d:Drug) ON (d.smiles)",
            "CREATE INDEX drug_category IF NOT EXISTS FOR (d:Drug) ON (d.category)",
            "CREATE INDEX interaction_severity IF NOT EXISTS FOR ()-[i:INTERACTS_WITH]-() ON (i.severity)",
        ]
        
        for query in queries:
            try:
                cls.run_query(query)
                logger.info(f"Executed: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Schema query failed (may already exist): {e}")
        
        return True
    
    # ========== Drug Operations ==========
    
    @classmethod
    def add_drug(cls, drugbank_id: str, name: str, smiles: str = None, 
                 category: str = None, description: str = None):
        """Add a drug node to the graph"""
        query = """
        MERGE (d:Drug {drugbank_id: $drugbank_id})
        SET d.name = $name,
            d.smiles = $smiles,
            d.category = $category,
            d.description = $description,
            d.updated_at = datetime()
        RETURN d
        """
        return cls.run_query(query, {
            'drugbank_id': drugbank_id,
            'name': name,
            'smiles': smiles,
            'category': category,
            'description': description
        })
    
    @classmethod
    def get_drug(cls, name: str = None, drugbank_id: str = None):
        """Get a drug by name or DrugBank ID"""
        if drugbank_id:
            query = "MATCH (d:Drug {drugbank_id: $id}) RETURN d"
            return cls.run_query(query, {'id': drugbank_id})
        elif name:
            query = "MATCH (d:Drug) WHERE toLower(d.name) CONTAINS toLower($name) RETURN d LIMIT 10"
            return cls.run_query(query, {'name': name})
        return []
    
    @classmethod
    def search_drugs(cls, query: str, limit: int = 20):
        """Search for drugs by name (fuzzy match)"""
        cypher = """
        MATCH (d:Drug)
        WHERE toLower(d.name) CONTAINS toLower($query)
           OR toLower(d.drugbank_id) CONTAINS toLower($query)
        RETURN d.drugbank_id as id, d.name as name, d.smiles as smiles, d.category as category
        ORDER BY 
            CASE WHEN toLower(d.name) STARTS WITH toLower($query) THEN 0 ELSE 1 END,
            d.name
        LIMIT $limit
        """
        return cls.run_query(cypher, {'query': query, 'limit': limit})
    
    # ========== Interaction Operations ==========
    
    @classmethod
    def add_interaction(cls, drug1_id: str, drug2_id: str, severity: str,
                       description: str = None, mechanism: str = None,
                       evidence_level: str = None):
        """Add a drug-drug interaction relationship"""
        query = """
        MATCH (d1:Drug {drugbank_id: $drug1_id})
        MATCH (d2:Drug {drugbank_id: $drug2_id})
        MERGE (d1)-[i:INTERACTS_WITH]->(d2)
        SET i.severity = $severity,
            i.description = $description,
            i.mechanism = $mechanism,
            i.evidence_level = $evidence_level,
            i.updated_at = datetime()
        RETURN d1, i, d2
        """
        return cls.run_query(query, {
            'drug1_id': drug1_id,
            'drug2_id': drug2_id,
            'severity': severity,
            'description': description,
            'mechanism': mechanism,
            'evidence_level': evidence_level
        })
    
    @classmethod
    def get_interactions(cls, drug_id: str):
        """Get all interactions for a drug"""
        query = """
        MATCH (d:Drug {drugbank_id: $drug_id})-[i:INTERACTS_WITH]-(other:Drug)
        RETURN other.drugbank_id as drug_id, 
               other.name as drug_name,
               i.severity as severity,
               i.description as description,
               i.mechanism as mechanism
        ORDER BY 
            CASE i.severity 
                WHEN 'severe' THEN 1 
                WHEN 'moderate' THEN 2 
                WHEN 'minor' THEN 3 
                ELSE 4 
            END
        """
        return cls.run_query(query, {'drug_id': drug_id})
    
    @classmethod
    def check_interaction(cls, drug1_id: str, drug2_id: str):
        """Check if two drugs have a known interaction"""
        query = """
        MATCH (d1:Drug {drugbank_id: $drug1_id})-[i:INTERACTS_WITH]-(d2:Drug {drugbank_id: $drug2_id})
        RETURN i.severity as severity,
               i.description as description,
               i.mechanism as mechanism,
               i.evidence_level as evidence_level
        LIMIT 1
        """
        results = cls.run_query(query, {'drug1_id': drug1_id, 'drug2_id': drug2_id})
        return results[0] if results else None
    
    # ========== Target Operations ==========
    
    @classmethod
    def add_target(cls, uniprot_id: str, name: str, gene_name: str = None,
                   organism: str = "Homo sapiens"):
        """Add a protein target node"""
        query = """
        MERGE (t:Target {uniprot_id: $uniprot_id})
        SET t.name = $name,
            t.gene_name = $gene_name,
            t.organism = $organism
        RETURN t
        """
        return cls.run_query(query, {
            'uniprot_id': uniprot_id,
            'name': name,
            'gene_name': gene_name,
            'organism': organism
        })
    
    @classmethod
    def link_drug_target(cls, drug_id: str, target_id: str, action: str = None,
                        known_action: bool = True):
        """Link a drug to its target protein"""
        query = """
        MATCH (d:Drug {drugbank_id: $drug_id})
        MATCH (t:Target {uniprot_id: $target_id})
        MERGE (d)-[r:TARGETS]->(t)
        SET r.action = $action,
            r.known_action = $known_action
        RETURN d, r, t
        """
        return cls.run_query(query, {
            'drug_id': drug_id,
            'target_id': target_id,
            'action': action,
            'known_action': known_action
        })
    
    # ========== Side Effect Operations ==========
    
    @classmethod
    def add_side_effect(cls, umls_id: str, name: str, organ_system: str = None):
        """Add a side effect node"""
        query = """
        MERGE (s:SideEffect {umls_id: $umls_id})
        SET s.name = $name,
            s.organ_system = $organ_system
        RETURN s
        """
        return cls.run_query(query, {
            'umls_id': umls_id,
            'name': name,
            'organ_system': organ_system
        })
    
    @classmethod
    def link_drug_side_effect(cls, drug_id: str, side_effect_id: str, 
                              frequency: str = None, severity: str = None):
        """Link a drug to a side effect"""
        query = """
        MATCH (d:Drug {drugbank_id: $drug_id})
        MATCH (s:SideEffect {umls_id: $side_effect_id})
        MERGE (d)-[r:CAUSES]->(s)
        SET r.frequency = $frequency,
            r.severity = $severity
        RETURN d, r, s
        """
        return cls.run_query(query, {
            'drug_id': drug_id,
            'side_effect_id': side_effect_id,
            'frequency': frequency,
            'severity': severity
        })
    
    # ========== Graph Analytics ==========
    
    @classmethod
    def get_common_targets(cls, drug1_id: str, drug2_id: str):
        """Find common protein targets between two drugs"""
        query = """
        MATCH (d1:Drug {drugbank_id: $drug1_id})-[:TARGETS]->(t:Target)<-[:TARGETS]-(d2:Drug {drugbank_id: $drug2_id})
        RETURN t.uniprot_id as target_id,
               t.name as target_name,
               t.gene_name as gene_name
        """
        return cls.run_query(query, {'drug1_id': drug1_id, 'drug2_id': drug2_id})
    
    @classmethod
    def get_interaction_path(cls, drug1_id: str, drug2_id: str, max_hops: int = 3):
        """Find interaction paths between two drugs"""
        query = """
        MATCH path = shortestPath(
            (d1:Drug {drugbank_id: $drug1_id})-[*1..$max_hops]-(d2:Drug {drugbank_id: $drug2_id})
        )
        RETURN [n in nodes(path) | labels(n)[0] + ': ' + coalesce(n.name, n.drugbank_id, n.uniprot_id)] as path_nodes,
               length(path) as path_length
        LIMIT 5
        """
        return cls.run_query(query, {
            'drug1_id': drug1_id, 
            'drug2_id': drug2_id,
            'max_hops': max_hops
        })
    
    @classmethod
    def get_drug_neighborhood(cls, drug_id: str, depth: int = 2):
        """Get the neighborhood graph around a drug for visualization"""
        query = """
        MATCH (d:Drug {drugbank_id: $drug_id})
        CALL apoc.path.subgraphAll(d, {
            maxLevel: $depth,
            relationshipFilter: 'INTERACTS_WITH|TARGETS|CAUSES'
        }) YIELD nodes, relationships
        RETURN 
            [n in nodes | {
                id: coalesce(n.drugbank_id, n.uniprot_id, n.umls_id),
                label: n.name,
                type: labels(n)[0]
            }] as nodes,
            [r in relationships | {
                source: coalesce(startNode(r).drugbank_id, startNode(r).uniprot_id, startNode(r).umls_id),
                target: coalesce(endNode(r).drugbank_id, endNode(r).uniprot_id, endNode(r).umls_id),
                type: type(r)
            }] as edges
        """
        return cls.run_query(query, {'drug_id': drug_id, 'depth': depth})
    
    @classmethod
    def get_polypharmacy_risks(cls, drug_ids: list):
        """Analyze risks for a combination of multiple drugs"""
        query = """
        UNWIND $drug_ids as drug_id
        MATCH (d:Drug {drugbank_id: drug_id})
        WITH collect(d) as drugs
        UNWIND drugs as d1
        UNWIND drugs as d2
        WITH d1, d2 WHERE id(d1) < id(d2)
        OPTIONAL MATCH (d1)-[i:INTERACTS_WITH]-(d2)
        RETURN d1.name as drug1, d2.name as drug2, 
               i.severity as severity, 
               i.mechanism as mechanism,
               i.description as description
        ORDER BY 
            CASE i.severity 
                WHEN 'severe' THEN 1 
                WHEN 'moderate' THEN 2 
                WHEN 'minor' THEN 3 
                ELSE 4 
            END
        """
        return cls.run_query(query, {'drug_ids': drug_ids})
    
    # ========== Statistics ==========
    
    @classmethod
    def get_stats(cls):
        """Get knowledge graph statistics"""
        query = """
        MATCH (d:Drug) WITH count(d) as drug_count
        MATCH (t:Target) WITH drug_count, count(t) as target_count
        MATCH (s:SideEffect) WITH drug_count, target_count, count(s) as side_effect_count
        MATCH ()-[i:INTERACTS_WITH]->() WITH drug_count, target_count, side_effect_count, count(i) as interaction_count
        RETURN drug_count, target_count, side_effect_count, interaction_count
        """
        results = cls.run_query(query)
        return results[0] if results else {
            'drug_count': 0,
            'target_count': 0,
            'side_effect_count': 0,
            'interaction_count': 0
        }
