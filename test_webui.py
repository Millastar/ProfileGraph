import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from langchain_core.messages import AIMessageChunk

import webUI


class FakeGraph:
    def stream(self, *_args, **_kwargs):
        metadata = {"langgraph_node": "generate"}
        yield AIMessageChunk(content="流式"), metadata
        yield AIMessageChunk(content="回复"), metadata


class WebUiTests(unittest.TestCase):
    def test_access_code_is_required_and_compared_exactly(self):
        with patch.dict(os.environ, {"APP_ACCESS_CODE": "demo-code"}, clear=False):
            failed = webUI.verify_access_code("wrong", 0)
            passed = webUI.verify_access_code("demo-code", 1)
        self.assertFalse(failed[0])
        self.assertEqual(failed[1], 1)
        self.assertTrue(passed[0])
        self.assertEqual(passed[1], 0)

    def test_conversation_sessions_are_independent(self):
        first, first_id = webUI._initial_conversations()
        second, second_id = webUI._initial_conversations()
        first[first_id]["messages"].append({"role": "user", "content": "only first"})
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(second[second_id]["messages"], [])

    def test_utf8_txt_is_chunked_without_persistence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "资料.txt"
            path.write_text("这是用于测试的中文资料。" * 120, encoding="utf-8-sig")
            chunks = webUI._extract_txt(path, path.name)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["filename"] == "资料.txt" for chunk in chunks))
        self.assertTrue(all(chunk["page"] is None for chunk in chunks))

    def test_cosine_ranking_is_scoped_to_supplied_documents(self):
        documents = [
            {
                "vectors": webUI._normalize_rows([[1.0, 0.0], [0.0, 1.0]]),
                "chunks": [
                    {"filename": "a.txt", "page": None, "content": "苹果"},
                    {"filename": "a.txt", "page": None, "content": "香蕉"},
                ],
            }
        ]
        ranked = webUI._rank_chunks([0.9, 0.1], documents, top_k=1)
        self.assertEqual(ranked[0][1]["content"], "苹果")

    def test_document_delete_removes_vectors_and_metadata(self):
        documents = [
            {"id": "a", "filename": "a.txt", "chunks": [{}], "vectors": np.ones((1, 2))},
            {"id": "b", "filename": "b.txt", "chunks": [{}], "vectors": np.ones((1, 2))},
        ]
        kept, _component_update, _status = webUI.delete_document("a", documents)
        self.assertEqual([document["id"] for document in kept], ["b"])

    def test_send_message_yields_each_model_chunk(self):
        conversations, conversation_id = webUI._initial_conversations()
        with patch.object(webUI, "_build_rag_graph", return_value=FakeGraph()):
            outputs = list(
                webUI.send_message("你好", conversations, conversation_id, "session-key", [])
            )
        histories = [output[0] for output in outputs]
        self.assertTrue(all(len(output) == 8 for output in outputs))
        self.assertEqual(histories[0], [{"role": "user", "content": "你好"}])
        self.assertEqual(histories[1][-1]["content"], "流式")
        self.assertEqual(histories[2][-1]["content"], "流式回复")
        self.assertEqual(histories[-1][-1]["content"], "流式回复")


if __name__ == "__main__":
    unittest.main()
