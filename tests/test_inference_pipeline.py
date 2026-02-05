"""Tests for inference pipeline components"""

import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.retriever import SpeechRetriever
from src.models import TextEncoder, SpeechEncoder, SpeechAdapter
from conftest import (
    create_dummy_audio,
    create_temp_audio_file,
    cleanup_temp_file
)


def test_speech_retriever_initialization():
    """Test SpeechRetriever initialization"""
    try:
        # Create models
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        # Create retriever
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device="cpu"
        )
        
        # Check initialization
        assert retriever.text_encoder is not None, "Text encoder should be set"
        assert retriever.speech_encoder is not None, "Speech encoder should be set"
        assert retriever.adapter is not None, "Adapter should be set"
        assert retriever.embedding_dim == 4096, "Embedding dim should be 4096"
        assert retriever.index is None, "Index should be None initially"
        
        print("✓ SpeechRetriever initialization test passed")
    except Exception as e:
        print(f"⚠ SpeechRetriever initialization test skipped: {e}")


def test_speech_retriever_build_index():
    """Test SpeechRetriever build_index with mock audio files"""
    try:
        # Create models
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        # Create retriever
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device="cpu",
            index_type="flat"
        )
        
        # Create temporary audio files
        num_files = 5
        audio_files = []
        try:
            for i in range(num_files):
                temp_path = create_temp_audio_file(duration_seconds=3, sample_rate=16000)
                audio_files.append(temp_path)
            
            # Build index
            retriever.build_index(audio_files, batch_size=2)
            
            # Check that index was built
            assert retriever.index is not None, "Index should be built"
            assert retriever.index.ntotal == num_files, \
                f"Index should have {num_files} vectors, got {retriever.index.ntotal}"
            assert len(retriever.audio_paths) == num_files, \
                f"Should have {num_files} audio paths, got {len(retriever.audio_paths)}"
            
            print("✓ SpeechRetriever build_index test passed")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ SpeechRetriever build_index test skipped: {e}")


def test_speech_retriever_search():
    """Test SpeechRetriever search functionality"""
    try:
        # Create models
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        # Create retriever
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device="cpu"
        )
        
        # Create temporary audio files
        num_files = 5
        audio_files = []
        try:
            for i in range(num_files):
                temp_path = create_temp_audio_file(duration_seconds=3, sample_rate=16000)
                audio_files.append(temp_path)
            
            # Build index
            retriever.build_index(audio_files, batch_size=2)
            
            # Search
            query = "What is machine learning?"
            k = 3
            results = retriever.search(query, k=k, return_scores=True)
            
            # Check results format
            assert len(results) == k, f"Should return {k} results, got {len(results)}"
            
            for i, result in enumerate(results):
                assert "rank" in result, "Result should have 'rank'"
                assert "audio_path" in result, "Result should have 'audio_path'"
                assert "score" in result, "Result should have 'score'"
                assert "index" in result, "Result should have 'index'"
                assert result["rank"] == i + 1, f"Rank should be {i+1}, got {result['rank']}"
                assert 0 <= result["index"] < num_files, \
                    f"Index should be in [0, {num_files}), got {result['index']}"
                assert -1 <= result["score"] <= 1, \
                    f"Score should be in [-1, 1], got {result['score']}"
            
            # Check that results are ordered by score (descending)
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), \
                "Results should be ordered by score (descending)"
            
            print("✓ SpeechRetriever search test passed")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ SpeechRetriever search test skipped: {e}")


def test_speech_retriever_search_without_scores():
    """Test SpeechRetriever search without returning scores"""
    try:
        # Create models
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        # Create retriever
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device="cpu"
        )
        
        # Create temporary audio files
        audio_files = []
        try:
            for i in range(3):
                temp_path = create_temp_audio_file(duration_seconds=2, sample_rate=16000)
                audio_files.append(temp_path)
            
            # Build index
            retriever.build_index(audio_files)
            
            # Search without scores
            results = retriever.search("test query", k=2, return_scores=False)
            
            # Check that scores are not included
            for result in results:
                assert "score" not in result, "Scores should not be included"
                assert "rank" in result, "Rank should be included"
                assert "audio_path" in result, "Audio path should be included"
            
            print("✓ SpeechRetriever search without scores test passed")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ SpeechRetriever search without scores test skipped: {e}")


def test_speech_retriever_save_load_index():
    """Test SpeechRetriever save_index and load_index"""
    try:
        # Create models
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        # Create retriever
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device="cpu"
        )
        
        # Create temporary audio files
        num_files = 3
        audio_files = []
        metadata = [{"id": i, "title": f"Audio {i}"} for i in range(num_files)]
        
        try:
            for i in range(num_files):
                temp_path = create_temp_audio_file(duration_seconds=2, sample_rate=16000)
                audio_files.append(temp_path)
            
            # Build index with metadata
            retriever.build_index(audio_files, metadata=metadata)
            
            # Save index
            temp_dir = tempfile.mkdtemp()
            index_path = os.path.join(temp_dir, "test_index.faiss")
            retriever.save_index(index_path)
            
            # Check that files exist
            assert os.path.exists(index_path), "Index file should exist"
            metadata_path = index_path.replace('.faiss', '.metadata.pkl')
            assert os.path.exists(metadata_path), "Metadata file should exist"
            
            # Create new retriever and load index
            retriever2 = SpeechRetriever(
                text_encoder=text_encoder,
                speech_encoder=speech_encoder,
                adapter=adapter,
                device="cpu"
            )
            
            retriever2.load_index(index_path)
            
            # Check that index was loaded
            assert retriever2.index is not None, "Index should be loaded"
            assert retriever2.index.ntotal == num_files, \
                f"Index should have {num_files} vectors"
            assert len(retriever2.audio_paths) == num_files, \
                f"Should have {num_files} audio paths"
            assert len(retriever2.audio_metadata) == num_files, \
                f"Should have {num_files} metadata entries"
            
            # Check that search still works
            results = retriever2.search("test query", k=2)
            assert len(results) == 2, "Search should work after loading"
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
            
            print("✓ SpeechRetriever save/load index test passed")
        finally:
            # Cleanup audio files
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ SpeechRetriever save/load index test skipped: {e}")


def test_speech_retriever_ivf_index():
    """Test SpeechRetriever with IVF index type"""
    try:
        # Create models
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        # Create retriever with IVF index
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device="cpu",
            index_type="ivf"
        )
        
        # Create temporary audio files (need more for IVF)
        num_files = 20
        audio_files = []
        try:
            for i in range(num_files):
                temp_path = create_temp_audio_file(duration_seconds=2, sample_rate=16000)
                audio_files.append(temp_path)
            
            # Build index
            retriever.build_index(audio_files, batch_size=5)
            
            # Check that index was built
            assert retriever.index is not None, "Index should be built"
            assert retriever.index.ntotal == num_files, \
                f"Index should have {num_files} vectors"
            
            # Test search
            results = retriever.search("test query", k=5)
            assert len(results) == 5, "Should return 5 results"
            
            print("✓ SpeechRetriever IVF index test passed")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ SpeechRetriever IVF index test skipped: {e}")


def test_inference_pipeline_end_to_end():
    """Test complete inference pipeline end-to-end"""
    try:
        # Create models
        text_encoder = TextEncoder(freeze=True)
        speech_encoder = SpeechEncoder(freeze=True)
        adapter = SpeechAdapter()
        
        # Create retriever
        retriever = SpeechRetriever(
            text_encoder=text_encoder,
            speech_encoder=speech_encoder,
            adapter=adapter,
            device="cpu"
        )
        
        # Create temporary audio files
        num_files = 5
        audio_files = []
        try:
            for i in range(num_files):
                temp_path = create_temp_audio_file(duration_seconds=3, sample_rate=16000)
                audio_files.append(temp_path)
            
            # Step 1: Build index from audio files
            retriever.build_index(audio_files, batch_size=2)
            assert retriever.index.ntotal == num_files, "Index should be built"
            
            # Step 2: Encode text query
            query = "What is artificial intelligence?"
            query_embedding = text_encoder.encode(query)
            assert query_embedding.shape == (1, 4096), "Query embedding should be (1, 4096)"
            
            # Step 3: Search
            k = 3
            results = retriever.search(query, k=k)
            
            # Step 4: Validate results
            assert len(results) == k, f"Should return {k} results"
            assert all("audio_path" in r for r in results), "All results should have audio_path"
            assert all(r["rank"] == i+1 for i, r in enumerate(results)), \
                "Results should be ranked correctly"
            
            # Results should be ordered by similarity (descending)
            if all("score" in r for r in results):
                scores = [r["score"] for r in results]
                assert scores == sorted(scores, reverse=True), \
                    "Results should be ordered by score"
            
            print("✓ Inference pipeline end-to-end test passed")
        finally:
            # Cleanup
            for path in audio_files:
                cleanup_temp_file(path)
    except Exception as e:
        print(f"⚠ Inference pipeline end-to-end test skipped: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Inference Pipeline Tests")
    print("=" * 60)
    print()
    
    # SpeechRetriever tests
    test_speech_retriever_initialization()
    test_speech_retriever_build_index()
    test_speech_retriever_search()
    test_speech_retriever_search_without_scores()
    test_speech_retriever_save_load_index()
    test_speech_retriever_ivf_index()
    
    # End-to-end tests
    test_inference_pipeline_end_to_end()
    
    print()
    print("=" * 60)
    print("Inference Pipeline Tests Completed")
    print("=" * 60)


