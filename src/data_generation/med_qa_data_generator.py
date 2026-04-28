import sys
import json
from typing import List, Dict, Any
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from providers import LLMProvider
from config.settings import settings
from knowledge_base.milvus.milvus_client import MilvusClient
from knowledge_base.lda.lda_pipeline import LDAPipeline
from data_generation.qa_corrector import QACorrector
from data_generation.evidence_based_verifier import EvidenceBasedVerifier


class MedQaDataGenerator:
    """医疗问答数据生成器

    流程:
    1. 对 Milvus 中的若干个 collection:
        - 对其中的每个 collection，首先进行主题建模，得到 num_topics 个主题（父主题）
        - 对每个父主题下面的所有文档，再逐一进行主题建模，得到子主题（sub-topic）
        - 对每个子主题，根据 MBTI 16 人格类型，生成对应的医生和患者背景资料
        - 对每个子主题 + 人格的组合，生成医患对话（第一轮必须是患者问）
            - 每轮对话前，先用 retrive_subtopic_documents 从 collection 中选出 top-k 片段
            - 每生成完一轮对话（医生或患者），进行循证验证（EvidenceBasedVerifier）
            - 验证不通过则使用 QACorrector 修复，再重新验证；通过则继续下一轮
    2. 对所有 collections 都执行上述步骤
    3. 合并所有 collection 对应的样本
    4. 将所有样本格式转换为 OpenAI 的 messages 格式（role:..., content:...）
    """

    def __init__(
        self,
        milvus_config=None,
        llm_config=None,
        save_path: str = None,
        data_num: int = 1000,
        textbook_collection_name: str = None,
    ):
        """
        初始化问答数据生成器

        Args:
            milvus_config: Milvus配置
            llm_config: LLM配置（可选，默认使用settings.llm）
            save_path: 数据保存路径
            data_num: 生成的问答数据数量
            textbook_collection_name: 教科书collection名称
        """
        self.data_num = data_num
        self.llm_config = llm_config if llm_config is not None else settings.llm
        self.llm_provider = LLMProvider(
            provider=str(self.llm_config.provider),
            model_name=str(self.llm_config.model_name),
            base_url=str(self.llm_config.base_url),
            api_key=str(self.llm_config.api_key),
            max_tokens=int(getattr(self.llm_config, 'max_tokens', 2048)),
            temperature=float(getattr(self.llm_config, 'temperature', 0.7)),
        )
        self.save_path = save_path
        self.textbook_collection_name = textbook_collection_name

        self.mbti_personality_types = [
            "ISTJ", "ISFJ", "INFJ", "INTJ",
            "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP",
            "ESTJ", "ESFJ", "ENFJ", "ENTJ",
        ]

        milvus_cfg = milvus_config if milvus_config is not None else settings.milvus
        self.milvus_client = MilvusClient(milvus_cfg)
        self.llm = self.llm_provider

        self.lda_pipeline = LDAPipeline(settings=settings)
        self.verifier = EvidenceBasedVerifier(self.llm_provider, self.milvus_client)
        self.corrector = QACorrector(self.llm_provider)

        self.doctor_gen_prompt = None
        self.patient_gen_prompt = None

    # ------------------------------------------------------------------
    # 角色背景资料生成
    # ------------------------------------------------------------------

    def generate_doctor(self, subtopic: str, mbti: str = "INTJ") -> str:
        """
        生成医生的背景资料

        1. 使用 retriever 在 milvus 中搜索 top-k 与 subtopic 相关的文档
        2. 使用 LLM 生成医生的背景资料，将文档段、MBTI 人格类型作为 context
        """
        docs = []
        if self.textbook_collection_name:
            docs = self.milvus_client.similarity_search(
                collection_name=self.textbook_collection_name,
                query=subtopic,
                k=3,
            )
        context = "\n".join(d.page_content for d in docs) if docs else "无参考文献"

        prompt = f"""请根据以下医学主题和 MBTI 人格类型，为一名医生创建详细的背景资料。

            ## 医学主题
            {subtopic}

            ## 医生性格（MBTI）
            {mbti}

            ## 参考文献摘要
            {context}

            ## 背景资料要求
            - 姓名（中文）
            - 职称与专科
            - 工作年限
            - 性格特点（符合 {mbti}）
            - 沟通风格
            - 擅长领域

            请输出医生背景资料："""

        return self.llm.generate(prompt).strip()



    def generate_patient(self, subtopic: str, mbti: str = "ISFP") -> str:
        """
        生成患者的背景资料

        结合相关文档片段和 MBTI 16 人格来生成患者背景资料

        1. 使用 retriever 在 milvus 中搜索 top-k 与 subtopic 相关的文档
        2. 调用 LLM 生成患者的背景资料
        """
        docs = []
        if self.textbook_collection_name:
            docs = self.milvus_client.similarity_search(
                collection_name=self.textbook_collection_name,
                query=subtopic,
                k=3,
            )
        context = "\n".join(d.page_content for d in docs) if docs else "无参考文献"

        prompt = f"""请根据以下医学主题和 MBTI 人格类型，为一名患者创建详细的背景资料。

            ## 医学主题
            {subtopic}

            ## 患者性格（MBTI）
            {mbti}

            ## 参考文献摘要
            {context}

            ## 背景资料要求
            - 姓名（中文）
            - 年龄、性别
            - 主诉（与主题相关的症状）
            - 病史摘要
            - 性格特点（符合 {mbti}）
            - 就医态度与沟通方式

            请输出患者背景资料："""

        return self.llm.generate(prompt).strip()

    # ------------------------------------------------------------------
    # 文档加载与主题建模
    # ------------------------------------------------------------------

    def load_documents_from_milvus(self, collection_name: str) -> List[str]:
        """
        从 Milvus collection 中加载所有文档文本

        Args:
            collection_name: Milvus collection 名称

        Returns:
            文档文本列表
        """
        # 使用一段通用查询来拉取较多文档
        docs = self.milvus_client.similarity_search(
            collection_name=collection_name,
            query="医疗 临床 诊断 治疗",
            k=200,
        )
        texts = [d.page_content for d in docs if d.page_content.strip()]
        return texts

    def modeling_topics_using_lda(
        self, documents: List[str], num_topics: int = 5
    ) -> Dict[str, Any]:
        """
        使用 LDAPipeline 对文档列表进行主题建模

        Args:
            documents: 文档文本列表
            num_topics: 期望主题数

        Returns:
            LDAPipeline.modeling_documents 返回的结果字典
        """
        result = self.lda_pipeline.modeling_documents(
            documents=documents,
            num_topics=num_topics,
            min_topic_size=2,
            top_n_words=10,
        )
        return result

    def generate_qa_topics(self, num_topics: int = 5) -> List[Dict[str, Any]]:
        """
        生成问答主题

        Args:
            num_topics: 生成的医疗主题数量

        Returns:
            List[Dict]: 包含主题ID和名称的列表
        """
        prompt = f"""请生成 {num_topics} 个不同的医疗问答主题，每个主题应覆盖临床诊疗的不同方面。

            ## 要求
            - 主题应来自不同临床科室（如内科、外科、妇产科、儿科等）
            - 每个主题应包含 id（整数）和 name（简洁的主题名称）
            - 输出 JSON 列表格式

            ## 输出示例
            [
            {{"id": 0, "name": "高血压的诊断与治疗"}},
            {{"id": 1, "name": "糖尿病并发症管理"}}
            ]

            请输出 {num_topics} 个主题："""

        response = self.llm.generate(prompt)
        try:
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                topics = json.loads(match.group())
                return topics[:num_topics]
        except Exception:
            pass

        # 解析失败时返回默认主题
        return [{"id": i, "name": f"医疗主题_{i}"} for i in range(num_topics)]

    def generate_subtopics(
        self, topic: str, num_subtopics: int = 3
    ) -> List[Dict[str, Any]]:
        """
        为单个父主题生成子主题

        Args:
            topic: 父主题名称
            num_subtopics: 生成的子主题数量

        Returns:
            List[Dict]: 包含子主题ID和名称的列表
        """
        prompt = f"""请为以下医疗主题生成 {num_subtopics} 个子主题。

            ## 父主题
            {topic}

            ## 要求
            - 每个子主题应聚焦于父主题的某个具体方面（如病因、诊断、治疗方案、护理等）
            - 输出 JSON 列表格式，每项包含 id（整数）和 name（简洁的子主题名称）

            请输出子主题列表："""

        response = self.llm.generate(prompt)
        try:
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                subtopics = json.loads(match.group())
                return subtopics[:num_subtopics]
        except Exception:
            pass

        return [{"id": i, "name": f"{topic}_子主题_{i}"} for i in range(num_subtopics)]


    def retrive_subtopic_documents(
        self, subtopic: str, collection_name: str, top_k: int = 5
    ) -> List[str]:
        """
        检索与子主题相关的文档片段

        Args:
            subtopic: 子主题名称（用作检索 query）
            collection_name: Milvus collection 名称
            top_k: 返回的文档片段数量

        Returns:
            文档内容列表
        """
        docs = self.milvus_client.similarity_search(
            collection_name=collection_name,
            query=subtopic,
            k=top_k,
        )
        return [d.page_content for d in docs if d.page_content.strip()]

    def _generate_one_turn(
        self,
        role: str,
        history: List[Dict[str, str]],
        doctor_profile: str,
        patient_profile: str,
        reference_docs: List[str],
        subtopic: str,
    ) -> str:
        """
        根据角色、历史对话和背景资料生成一轮对话内容

        Args:
            role: "doctor" 或 "patient"
            history: 已有对话历史
            doctor_profile: 医生背景资料
            patient_profile: 患者背景资料
            reference_docs: 当前检索到的参考文档
            subtopic: 子主题

        Returns:
            生成的对话内容字符串
        """
        history_text = "\n".join(
            f"{'医生' if t['role'] == 'doctor' else '患者'}: {t['content']}"
            for t in history
        )
        ref_text = "\n".join(reference_docs[:10]) if reference_docs else "无参考文献"
        role_cn = "医生" if role == "doctor" else "患者"
        profile = doctor_profile if role == "doctor" else patient_profile

        prompt = f"""你正在扮演一名 {role_cn}，请根据背景资料和当前对话，生成下一轮 {role_cn} 的对话内容。

            ## {role_cn}背景资料
            {profile}

            ## 当前对话主题
            {subtopic}

            ## 参考医学文献
            {ref_text}

            ## 历史对话
            {history_text if history_text else '（对话刚开始）'}

            ## 要求
            - 内容符合 {role_cn} 的身份和性格
            - 表述清晰、贴近真实临床场景
            - 不超过 200 字
            - 只输出 {role_cn} 说的内容，不要加角色前缀

            请输出 {role_cn} 的对话："""

        return self.llm.generate(prompt).strip()


        

    def generate_doc_pat_conversation_foreach_subtopic(
        self,
        subtopic_docs: Dict[int, List[str]],
        subtopic_names: Dict[int, str],
        collection_name: str,
        max_turns: int = 6,
        max_retries: int = 3,
    ) -> Dict[int, List[Dict[str, str]]]:
        """
        为每个子主题生成医生-患者对话

        Args:
            subtopic_docs: {subtopic_id: [doc_text, ...]}
            subtopic_names: {subtopic_id: subtopic_name}
            collection_name: Milvus collection 名称
            max_turns: 对话最大轮次（一轮 = 医生或患者各说一句）
            max_retries: 循证验证失败后最大重试次数

        Returns:
            Dict: {subtopic_id: [{"role": ..., "content": ...}, ...]}
        """
        all_conversations: Dict[int, List[Dict[str, str]]] = {}

        for subtopic_id, docs in subtopic_docs.items():
            subtopic_name = subtopic_names.get(subtopic_id, f"子主题_{subtopic_id}")

            # 生成16条（每种MBTI人格组合一条）
            subtopic_conversations = []
            for mbti in self.mbti_personality_types:
                doctor_profile = self.generate_doctor(subtopic_name, mbti)
                # 从剩余的 mbti 中随机抽取一个人格， 生成患者背景资料
                remaining_mbti = [rmbti for rmbti in self.mbti_personality_types if rmbti != mbti]
                import random
                patient_mbti = random.choice(remaining_mbti)
                patient_profile = self.generate_patient(subtopic_name, patient_mbti)

                history: List[Dict[str, str]] = []
                # 第一轮必须是患者
                turn_roles = ["patient"] + ["doctor", "patient"] * (max_turns // 2)

                for role in turn_roles[:max_turns]:
                    ref_docs = self.retrive_subtopic_documents(
                        subtopic_name, collection_name, top_k=5
                    )
                    content = self._generate_one_turn(
                        role, history, doctor_profile, patient_profile,
                        ref_docs, subtopic_name,
                    )
                    current_turn = {"role": role, "content": content}

                    # 循证验证 + 修正循环
                    for attempt in range(max_retries):
                        verify_result = self.verifier.verify(
                            history, current_turn, ref_docs, collection_name
                        )
                        if verify_result["passed"]:
                            break
                        current_turn = self.corrector.correct(
                            history, current_turn, verify_result["feedback"]
                        )

                    history.append(current_turn)

                subtopic_conversations.append(history)

            all_conversations[subtopic_id] = subtopic_conversations

        return all_conversations

    def generate_qa_data(self) -> List[Dict[str, Any]]:
        """
        生成QA数据的主流程

        Returns:
            List[Dict]: 包含 OpenAI messages 格式问答数据的列表
        """
        all_qa_data: List[Dict[str, Any]] = []

        collection_names = [self.textbook_collection_name] if self.textbook_collection_name else []
        if not collection_names:
            collection_names = self.milvus_client.list_collections()

        for collection_name in collection_names:
            # 1. 加载文档
            documents = self.load_documents_from_milvus(collection_name)
            if not documents:
                continue

            # 2. 主题建模（父主题）
            lda_result = self.modeling_topics_using_lda(documents, num_topics=5)
            if "error" in lda_result:
                continue

            # 3. 为每个父主题生成子主题
            parent_topics = self.generate_qa_topics(num_topics=5)
            all_subtopic_docs: Dict[int, List[str]] = {}
            all_subtopic_names: Dict[int, str] = {}
            subtopic_offset = 0

            for topic in parent_topics:
                subtopics = self.generate_subtopics(topic["name"], num_subtopics=3)
                for sub in subtopics:
                    global_id = subtopic_offset + sub["id"]
                    all_subtopic_names[global_id] = sub["name"]
                    all_subtopic_docs[global_id] = self.retrive_subtopic_documents(
                        sub["name"], collection_name, top_k=10
                    )
                subtopic_offset += len(subtopics)

            # 4. 为每个子主题生成医患对话
            conversations = self.generate_doc_pat_conversation_foreach_subtopic(
                subtopic_docs=all_subtopic_docs,
                subtopic_names=all_subtopic_names,
                collection_name=collection_name,
            )

            # 5. 转换为 OpenAI messages 格式
            for subtopic_id, conv_list in conversations.items():
                for conv in conv_list:
                    messages = [
                        {
                            "role": "user" if t["role"] == "patient" else "assistant",
                            "content": t["content"],
                        }
                        for t in conv
                    ]
                    all_qa_data.append({
                        "collection": collection_name,
                        "subtopic": all_subtopic_names.get(subtopic_id, ""),
                        "messages": messages,
                    })

            if len(all_qa_data) >= self.data_num:
                break

        return all_qa_data[:self.data_num]

    def save_generated_qa_data(self, qa_data: List[Dict[str, Any]] = None) -> str:
        """
        将生成的 QA 数据保存到文件

        Args:
            qa_data: 待保存的数据

        Returns:
            保存文件的路径
        """
        if self.save_path is None:
            raise ValueError("save_path 未设置，请在初始化时指定 save_path 参数")

        if qa_data is None:
            raise ValueError("qa_data 未设置，请在初始化时指定 qa_data 参数")

        save_path = Path(self.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if not str(save_path).endswith(".json"):
            raise ValueError("save_path 必须以 .json 结尾, 目前我们只支持 JSON")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=2)

        print(f"QA 数据已保存到: {save_path}，共 {len(qa_data)} 条记录")
        return str(save_path)


def run():
    """运行医疗问答数据生成器示例（使用 settings.milvus 和 settings.llm）"""
    save_path = str(Path(__file__).parent.parent.parent / "data" / "generated_qa_data")

    generator = MedQaDataGenerator(
        llm_config=settings.llm,
        save_path=save_path,
        data_num=100,
        textbook_collection_name=None,
    )

    qa_data = generator.generate_qa_data()
    generator.save_generated_qa_data(qa_data)
    print(f"生成完成，共 {len(qa_data)} 条数据")


if __name__ == "__main__":
    run()
