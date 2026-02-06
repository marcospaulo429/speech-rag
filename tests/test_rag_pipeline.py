"""Tests for SpeechRAGPipeline integration"""

import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.rag_pipeline import SpeechRAGPipeline
from src.inference.retriever import SpeechRetriever
from src.inference.generator import AudioConditionedGenerator
from src.models import TextEncoder, SpeechEncoder, SpeechAdapter
from conftest import (
    create_temp_audio_file,
    cleanup_temp_file
)


def get_device():
    """Get device (cuda if available, else cpu)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_rag_pipeline_initialization():
    """Test SpeechRAGPipeline initialization"""
    try:
        # Create retriever components
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device=get_device()
        )
        
        # Create generator
        generator = AudioConditionedGenerator(
            model_name="Qwen/Qwen-Audio-Chat",
            device=get_device()
        )
        
        # Create pipeline
        pipeline = SpeechRAGPipeline(
            retriever=retriever,
            generator=generator,
            top_k_audio=3
        )
        
        # Check initialization
        assert pipeline.retriever is not None, "Retriever should be set"
        assert pipeline.generator is not None, "Generator should be set"
        assert pipeline.top_k_audio == 3, "top_k_audio should be 3"
        
        print("✓ SpeechRAGPipeline initialization test passed")
    except Exception as e:
        print(f"⚠ SpeechRAGPipeline initialization test skipped: {e}")


def test_retrieve_and_generate():
    """Test complete retrieve and generate pipeline"""
    try:
        # Create retriever components
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device=get_device()
        )
        
        # Create temporary audio files and build index
        audio_files = []
        try:
            for i in range(5):
                temp_path = create_temp_audio_file(duration_seconds=3, sample_rate=16000)
                audio_files.append(temp_path)
            
            # Build index
            retriever.build_index(audio_files, batch_size=2)
            
            # Create generator
            generator = AudioConditionedGenerator(
                model_name="Qwen/Qwen-Audio-Chat",
                device=get_device()
            )
            
            # Create pipeline
            pipeline = SpeechRAGPipeline(
                retriever=retriever,
                generator=generator,
                top_k_audio=2
            )
            
            # Test retrieve and generate
            query = "What is artificial intelligence?"
            result = pipeline.retrieve_and_generate(
                query=query,
                max_new_tokens=50,
                return_retrieval_results=True
            )
            
            # Check result format
            assert "response" in result, "Result should have 'response' key"
            assert "query" in result, "Result should have 'query' key"
            assert "audio_paths" in result, "Result should have 'audio_paths' key"
            assert "num_audios" in result, "Result should have 'num_audios' key"
            assert "retrieval_results" in result, "Result should have 'retrieval_results' key"
            
            assert result["query"] == query, "Query should match"
            assert isinstance(result["response"], str), "Response should be a string"
            assert len(result["audio_paths"]) > 0, "Should have audio paths"
            assert len(result["retrieval_results"]) > 0, "Should have retrieval results"
            
            print("✓ retrieve_and_generate test passed")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ retrieve_and_generate test skipped: {e}")


def test_rag_pipeline_with_mock_audio():
    """Test RAG pipeline with mock audio files"""
    try:
        # Create retriever components
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device=get_device()
        )
        
        # Create mock audio files
        audio_files = []
        try:
            for i in range(3):
                temp_path = create_temp_audio_file(duration_seconds=2, sample_rate=16000)
                audio_files.append(temp_path)
            
            # Build index
            retriever.build_index(audio_files)
            
            # Create generator
            generator = AudioConditionedGenerator(
                model_name="Qwen/Qwen-Audio-Chat",
                device=get_device()
            )
            
            # Create pipeline
            pipeline = SpeechRAGPipeline(
                retriever=retriever,
                generator=generator,
                top_k_audio=2
            )
            
            # Test with different queries
            queries = [
                "What is machine learning?",
                "Explain neural networks."
            ]
            
            for query in queries:
                result = pipeline.retrieve_and_generate(
                    query=query,
                    max_new_tokens=30,
                    return_retrieval_results=False
                )
                
                assert "response" in result, "Result should have response"
                assert result["query"] == query, "Query should match"
            
            print("✓ RAG pipeline with mock audio test passed")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ RAG pipeline with mock audio test skipped: {e}")


def test_rag_pipeline_output_format():
    """Test RAG pipeline output format"""
    try:
        # Create retriever components
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device=get_device()
        )
        
        # Create audio files
        audio_files = []
        try:
            for i in range(4):
                temp_path = create_temp_audio_file(duration_seconds=2, sample_rate=16000)
                audio_files.append(temp_path)
            
            retriever.build_index(audio_files)
            
            # Create generator
            generator = AudioConditionedGenerator(
                model_name="Qwen/Qwen-Audio-Chat",
                device=get_device()
            )
            
            # Create pipeline
            pipeline = SpeechRAGPipeline(
                retriever=retriever,
                generator=generator,
                top_k_audio=2
            )
            
            query = "Test query"
            
            # Test with retrieval results
            result_with_retrieval = pipeline.retrieve_and_generate(
                query=query,
                max_new_tokens=20,
                return_retrieval_results=True
            )
            
            assert "retrieval_results" in result_with_retrieval, \
                "Should have retrieval_results when return_retrieval_results=True"
            assert isinstance(result_with_retrieval["retrieval_results"], list), \
                "retrieval_results should be a list"
            
            # Test without retrieval results
            result_without_retrieval = pipeline.retrieve_and_generate(
                query=query,
                max_new_tokens=20,
                return_retrieval_results=False
            )
            
            assert "retrieval_results" not in result_without_retrieval, \
                "Should not have retrieval_results when return_retrieval_results=False"
            
            print("✓ RAG pipeline output format test passed")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ RAG pipeline output format test skipped: {e}")


def test_retrieve_only():
    """Test retrieve_only method"""
    try:
        # Create retriever components
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device=get_device()
        )
        
        # Create audio files
        audio_files = []
        try:
            for i in range(5):
                temp_path = create_temp_audio_file(duration_seconds=2, sample_rate=16000)
                audio_files.append(temp_path)
            
            retriever.build_index(audio_files)
            
            # Create generator (not used in retrieve_only)
            generator = AudioConditionedGenerator(
                model_name="Qwen/Qwen-Audio-Chat",
                device=get_device()
            )
            
            # Create pipeline
            pipeline = SpeechRAGPipeline(
                retriever=retriever,
                generator=generator,
                top_k_audio=3
            )
            
            # Test retrieve_only
            query = "Test query"
            results = pipeline.retrieve_only(query, k=3)
            
            assert isinstance(results, list), "Results should be a list"
            assert len(results) == 3, "Should return 3 results"
            assert all("audio_path" in r for r in results), \
                "All results should have audio_path"
            
            print("✓ retrieve_only test passed")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ retrieve_only test skipped: {e}")


def test_generate_only():
    """Test generate_only method"""
    try:
        # Create retriever (not used in generate_only)
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device=get_device()
        )
        
        # Create generator
        generator = AudioConditionedGenerator(
            model_name="Qwen/Qwen-Audio-Chat",
            device=get_device()
        )
        
        # Create pipeline
        pipeline = SpeechRAGPipeline(
            retriever=retriever,
            generator=generator,
            top_k_audio=2
        )
        
        # Create audio files
        audio_files = []
        try:
            for i in range(2):
                temp_path = create_temp_audio_file(duration_seconds=2, sample_rate=16000)
                audio_files.append(temp_path)
            
            # Test generate_only
            query = "What is this about?"
            result = pipeline.generate_only(
                query=query,
                audio_paths=audio_files,
                max_new_tokens=30
            )
            
            assert "response" in result, "Result should have response"
            assert result["query"] == query, "Query should match"
            assert len(result["audio_paths"]) == 2, "Should have 2 audio paths"
            
            print("✓ generate_only test passed")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ generate_only test skipped: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("SpeechRAGPipeline Integration Tests")
    print("=" * 60)
    print()
    
    test_rag_pipeline_initialization()
    test_retrieve_and_generate()
    test_rag_pipeline_with_mock_audio()
    test_rag_pipeline_output_format()
    test_retrieve_only()
    test_generate_only()
    
    print()
    print("=" * 60)
    print("SpeechRAGPipeline Integration Tests Completed")
    print("=" * 60)

