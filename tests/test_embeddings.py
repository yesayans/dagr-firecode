"""Embeddings: providers, disk cache, and the addressable matrix."""

from __future__ import annotations

import numpy as np
import pytest

from aipm.config import Settings
from aipm.embeddings.cache import EmbeddingCache
from aipm.embeddings.provider import (
    FixtureEmbeddingProvider,
    TfidfSvdProvider,
    build_embedding_provider,
    l2_normalise,
)
from aipm.embeddings.store import EmbeddingService
from aipm.schemas import ReviewUnit


def units(texts: list[str]) -> list[ReviewUnit]:
    return [
        ReviewUnit(unit_id=f"u{i}", review_id=f"r{i}", app_id="a1", text=t)
        for i, t in enumerate(texts)
    ]


class TestFixtureProvider:
    def test_deterministic_across_instances(self):
        a = FixtureEmbeddingProvider(dim=16).embed(["hello"])
        b = FixtureEmbeddingProvider(dim=16).embed(["hello"])
        np.testing.assert_allclose(a, b)

    def test_rows_are_unit_length(self):
        vectors = FixtureEmbeddingProvider(dim=16).embed(["a", "b", "c"])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)

    def test_declares_itself_non_semantic(self):
        """Hashed noise must not be trusted by the citation relevance guard."""
        assert FixtureEmbeddingProvider().supports_semantic_similarity is False

    def test_empty_input(self):
        assert FixtureEmbeddingProvider(dim=8).embed([]).shape == (0, 8)


class TestTfidfSvdProvider:
    def test_requires_fit_before_embed(self):
        with pytest.raises(RuntimeError, match="fit"):
            TfidfSvdProvider().embed(["anything"])

    def test_fit_signature_is_part_of_the_cache_namespace(self):
        """Vectors depend on the fitted corpus, so the model id must too."""
        corpus_a = [f"the app crashes on login attempt number {i}" for i in range(20)]
        corpus_b = [f"delivery driver could not find address {i}" for i in range(20)]
        a, b = TfidfSvdProvider(dim=4), TfidfSvdProvider(dim=4)
        a.fit(corpus_a)
        b.fit(corpus_b)
        assert a.model != b.model

    def test_declares_itself_non_semantic(self):
        assert TfidfSvdProvider().supports_semantic_similarity is False


class TestL2Normalise:
    def test_zero_vector_does_not_produce_nan(self):
        out = l2_normalise(np.zeros((1, 4), dtype=np.float32))
        assert not np.isnan(out).any()


class TestEmbeddingCache:
    def test_round_trip(self, tmp_path):
        cache = EmbeddingCache(tmp_path / "vectors.sqlite")
        vectors = np.random.default_rng(0).standard_normal((2, 8)).astype(np.float32)
        cache.put_many("m1", ["alpha", "beta"], vectors)
        got = cache.get_many("m1", ["alpha", "beta"])
        assert len(got) == 2

    def test_namespaced_by_model(self, tmp_path):
        cache = EmbeddingCache(tmp_path / "vectors.sqlite")
        cache.put_many("m1", ["alpha"], np.ones((1, 4), dtype=np.float32))
        assert cache.get_many("m2", ["alpha"]) == {}

    def test_normalised_text_hits_the_same_entry(self, tmp_path):
        """Trivial whitespace/case differences must not miss the cache."""
        cache = EmbeddingCache(tmp_path / "vectors.sqlite")
        cache.put_many("m1", ["Hello  World"], np.ones((1, 4), dtype=np.float32))
        assert cache.get_many("m1", ["hello world"])

    def test_length_mismatch_rejected(self, tmp_path):
        cache = EmbeddingCache(tmp_path / "vectors.sqlite")
        with pytest.raises(ValueError):
            cache.put_many("m1", ["a", "b"], np.ones((1, 4), dtype=np.float32))

    def test_survives_more_than_the_sqlite_parameter_limit(self, tmp_path):
        """get_many chunks its IN clause; 1500 keys would otherwise fail."""
        cache = EmbeddingCache(tmp_path / "vectors.sqlite")
        texts = [f"text number {i}" for i in range(1500)]
        cache.put_many("m1", texts, np.ones((1500, 4), dtype=np.float32))
        assert len(cache.get_many("m1", texts)) == 1500


class TestEmbeddingService:
    def test_row_order_matches_input(self, embedding_service):
        matrix = embedding_service.embed_units(units(["one", "two", "three"]))
        assert matrix.unit_ids == ["u0", "u1", "u2"]
        assert len(matrix) == 3

    def test_repeated_text_embedded_once(self, tmp_path):
        provider = FixtureEmbeddingProvider(dim=8)
        calls: list[int] = []
        original = provider.embed
        provider.embed = lambda texts: (calls.append(len(texts)), original(texts))[1]

        service = EmbeddingService(provider, EmbeddingCache(tmp_path / "c.sqlite"))
        service.embed_units(units(["same", "same", "same", "other"]))
        assert calls == [2]  # two unique texts, not four

    def test_cache_prevents_second_provider_call(self, tmp_path):
        cache = EmbeddingCache(tmp_path / "c.sqlite")
        texts = ["alpha", "beta"]
        EmbeddingService(FixtureEmbeddingProvider(dim=8), cache).embed_texts(texts)

        provider = FixtureEmbeddingProvider(dim=8)
        provider.embed = lambda texts: pytest.fail("provider called despite a warm cache")
        EmbeddingService(provider, cache).embed_texts(texts)

    def test_row_lookup_by_unit_id(self, embedding_service):
        matrix = embedding_service.embed_units(units(["one", "two"]))
        np.testing.assert_allclose(matrix.row("u1"), matrix.vectors[1])

    def test_empty_units(self, embedding_service):
        assert len(embedding_service.embed_units([])) == 0


class TestFactory:
    def test_fixture_backend_resolves_dim_from_settings(self):
        provider = build_embedding_provider(Settings(embed_backend="fixture", embed_dim=64))
        assert provider.dim == 64

    def test_api_backend_without_key_raises(self):
        with pytest.raises(ValueError, match="API_KEY"):
            build_embedding_provider(
                Settings(embed_backend="api", llm_api_key="", embed_api_key="")
            )
