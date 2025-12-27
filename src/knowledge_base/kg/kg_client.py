from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent))
import spacy
from config.kg_config import KGConfig
from typing import List, Dict, Any, Optional
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from ner.ner_service import NERService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGClient:
    """知识图谱客户端"""
    def __init__(self, config: Optional[KGConfig] = None, ner_service: Optional[NERService] = None):
        """初始化知识图谱客户端"""
        self.config = config or KGConfig()
        self.ner_service = ner_service or NERService()
        self.graph = None
        self._connect()

    def _connect(self) -> None:
        """连接到Neo4j数据库"""
        try:
            self.graph = Neo4jGraph(
                url=self.config.uri,
                username=self.config.username,
                password=self.config.password
            )
            logger.info(f"Successfully connected to Neo4j at {self.config.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索知识图谱中的实体"""
        try:
            cypher_query = """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($query)
            OR toLower(n.id) CONTAINS toLower($query)
            RETURN n, labels(n) as labels
            LIMIT $limit
            """
            results = self.graph.query(cypher_query, params={"query": query, "limit": limit})

            entities = []
            for record in results:
                node = record["n"]
                entity = dict(node)
                entity["labels"] = record["labels"]
                entities.append(entity)

            return entities
        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            return []

    def search_relations(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索知识图谱中的关系"""
        try:
            cypher_query = """
            MATCH ()-[r]->()
            WHERE toLower(type(r)) CONTAINS toLower($query)
            RETURN r, type(r) as relation_type
            LIMIT $limit
            """
            results = self.graph.query(cypher_query, params={"query": query, "limit": limit})

            relations = []
            for record in results:
                relation = dict(record["r"])
                relation["relation_type"] = record["relation_type"]
                relations.append(relation)

            return relations
        except Exception as e:
            logger.error(f"Error searching relations: {e}")
            return []

    def search_subgraphs(self, query: str, max_depth: int = 2, limit: int = 50) -> Dict[str, Any]:
        """搜索知识图谱中的子图"""
        try:
            entities = self.ner_service.extract_entities(query)
            if not entities:
                logger.warning("No entities found in query for subgraph search")
                return {"entities": [], "relations": [], "subgraph": {}}

            entity_texts = [entity["text"] for entity in entities]

            cypher_query = """
            MATCH (start)
            WHERE start.name IN $entity_texts
            CALL {
                WITH start
                MATCH path = (start)-[*1..$max_depth]-(connected)
                RETURN nodes(path) as nodes, relationships(path) as rels
                LIMIT $limit
            }
            RETURN nodes, rels
            """

            results = self.graph.query(
                cypher_query,
                params={
                    "entity_texts": entity_texts,
                    "max_depth": max_depth,
                    "limit": limit
                }
            )

            all_nodes = set()
            all_relations = []
            subgraph_edges = []

            for record in results:
                nodes = record["nodes"]
                rels = record["rels"]

                for node in nodes:
                    node_dict = dict(node)
                    node_dict["labels"] = list(node.labels) if hasattr(node, 'labels') else []
                    all_nodes.add(node_dict)

                for rel in rels:
                    rel_dict = {
                        "source": dict(rel.start_node)["name"] if hasattr(rel.start_node, 'name') else str(rel.start_node),
                        "target": dict(rel.end_node)["name"] if hasattr(rel.end_node, 'name') else str(rel.end_node),
                        "type": rel.type,
                        "properties": dict(rel)
                    }
                    all_relations.append(rel_dict)
                    subgraph_edges.append({
                        "source": rel_dict["source"],
                        "target": rel_dict["target"],
                        "relation": rel.type
                    })

            return {
                "entities": list(all_nodes),
                "relations": all_relations,
                "subgraph": {
                    "nodes": [{"id": n.get("name", str(n)), "labels": n.get("labels", [])} for n in all_nodes],
                    "edges": subgraph_edges
                },
                "extracted_entities": entities
            }
        except Exception as e:
            logger.error(f"Error searching subgraphs: {e}")
            return {"entities": [], "relations": [], "subgraph": {}, "error": str(e)}

    def entity_linking(self, query: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """实体链接：将文本中的实体指称关联到知识库中的实体"""
        try:
            entities = self.ner_service.extract_entities(query)

            if not entities:
                logger.info("No entities found in query")
                return []

            linked_entities = []

            for entity in entities:
                mention = entity["text"]
                label = entity.get("label", "")

                cypher_query = """
                MATCH (n)
                WHERE toLower(n.name) = toLower($mention)
                RETURN n, labels(n) as labels
                LIMIT 1
                """

                results = self.graph.query(cypher_query, params={"mention": mention})

                if results:
                    record = results[0]
                    node = record["n"]
                    linked_entity = {
                        "mention": mention,
                        "label": label,
                        "kg_entity": dict(node),
                        "kg_labels": record["labels"],
                        "linked": True,
                        "confidence": 1.0
                    }
                else:
                    linked_entity = {
                        "mention": mention,
                        "label": label,
                        "kg_entity": None,
                        "kg_labels": [],
                        "linked": False,
                        "confidence": 0.0
                    }

                linked_entities.append(linked_entity)

            return linked_entities

        except Exception as e:
            logger.error(f"Error in entity linking: {e}")
            return []

    def get_entity_neighbors(self, entity_name: str, relation_type: Optional[str] = None) -> Dict[str, Any]:
        """获取实体的邻居节点"""
        try:
            if relation_type:
                cypher_query = """
                MATCH (n {name: $entity_name})-[r:`{}`]->(m)
                RETURN n, r, m, type(r) as rel_type
                """.format(relation_type)
            else:
                cypher_query = """
                MATCH (n {name: $entity_name})-[r]->(m)
                RETURN n, r, m, type(r) as rel_type
                """

            results = self.graph.query(cypher_query, params={"entity_name": entity_name})

            neighbors = []
            relations = []

            for record in results:
                neighbor = dict(record["m"])
                neighbor["labels"] = list(record["m"].labels) if hasattr(record["m"], 'labels') else []

                relation = {
                    "type": record["rel_type"],
                    "properties": dict(record["r"]),
                    "neighbor": neighbor
                }
                relations.append(relation)
                neighbors.append(neighbor)

            return {
                "entity": entity_name,
                "neighbors": neighbors,
                "relations": relations,
                "neighbor_count": len(neighbors)
            }
        except Exception as e:
            logger.error(f"Error getting entity neighbors: {e}")
            return {"entity": entity_name, "neighbors": [], "relations": [], "neighbor_count": 0}

    def execute_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行自定义Cypher查询"""
        try:
            results = self.graph.query(query, params=params or {})
            return results
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            raise

    def close(self) -> None:
        """关闭数据库连接"""
        if self.graph:
            self.graph.close()
            logger.info("Neo4j connection closed")
