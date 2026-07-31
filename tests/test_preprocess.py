"""Preprocessing: cleaning, language, segmentation, dedupe, quality."""

from __future__ import annotations

from datetime import date

from aipm.preprocess.clean import clean_review_text, normalise_for_dedup
from aipm.preprocess.dedupe import find_duplicate_groups, jaccard, mark_duplicates
from aipm.preprocess.language import detect_language, is_english
from aipm.preprocess.pipeline import PreprocessConfig, ReviewPreprocessor
from aipm.preprocess.quality import is_pure_praise, lexical_diversity, quality_weight
from aipm.preprocess.segment import segment_review, split_text
from aipm.schemas import Review


def make_review(text: str, **kwargs) -> Review:
    defaults = {
        "review_id": "r1", "app_id": "a1", "score": 3,
        "review_date": date(2024, 1, 1), "helpful_count": 0,
    }
    defaults.update(kwargs)
    return Review(text=text, **defaults)


class TestClean:
    def test_strips_urls_and_emails(self):
        out = clean_review_text("see https://x.com or mail me@x.com now")
        assert "http" not in out and "@" not in out

    def test_collapses_elongation(self):
        assert clean_review_text("sooooo slow!!!!!") == "soo slow!!"

    def test_normalise_for_dedup_ignores_case_and_punctuation(self):
        assert normalise_for_dedup("Great app!") == normalise_for_dedup("great app")

    def test_empty_input(self):
        assert clean_review_text("") == ""


class TestLanguage:
    def test_english_accepted(self):
        assert is_english("the app keeps crashing when I open my inbox")

    def test_other_latin_language_rejected(self):
        assert not is_english(
            "la aplicacion se cierra sola cuando abro el correo electronico"
        )

    def test_non_latin_script_rejected(self):
        assert not is_english("アプリがクラッシュします")

    def test_short_text_is_kept(self):
        """Biased toward keeping: dropping real English costs us evidence."""
        assert is_english("crashes constantly")

    def test_detect_language_labels(self):
        assert detect_language("this is clearly an english sentence about the app") == "en"
        assert detect_language("アプリ") == "other"


class TestSegment:
    def test_splits_on_sentences(self):
        assert len(split_text("It crashes. Support never replied.")) == 2

    def test_splits_on_contrastive_connective(self):
        """One review, two distinct complaints - the core reason we segment."""
        parts = split_text("Love the design but it logs me out every single day")
        assert len(parts) == 2
        assert "logs me out" in parts[1]

    def test_short_fragments_dropped(self):
        units = segment_review(make_review("Good. It crashes on every single login attempt"),
                               min_tokens=4)
        assert len(units) == 1

    def test_short_review_kept_whole_rather_than_lost(self):
        units = segment_review(make_review("crashes constantly"), min_tokens=4)
        assert len(units) == 1

    def test_unit_ids_are_stable_across_calls(self):
        review = make_review("It crashes on login. Payment also fails every time.")
        assert [u.unit_id for u in segment_review(review)] == [
            u.unit_id for u in segment_review(review)
        ]

    def test_units_retain_review_id(self):
        units = segment_review(make_review("It crashes on login. Payment fails too.",
                                           review_id="rX"))
        assert all(u.review_id == "rX" for u in units)


class TestDedupe:
    def test_exact_duplicates_grouped(self):
        groups = find_duplicate_groups(["Great app!", "great app", "totally different text"])
        assert groups[1] == groups[0]
        assert groups[2] == 2

    def test_near_duplicates_grouped(self):
        groups = find_duplicate_groups(
            ["the app is very good and fast", "the app is very good and fast!!"],
            threshold=0.8,
        )
        assert groups[1] == groups[0]

    def test_jaccard_bounds(self):
        assert jaccard(set(), {"a"}) == 0.0
        assert jaccard({"a"}, {"a"}) == 1.0

    def test_mark_duplicates_flags_second_occurrence_only(self):
        marked = mark_duplicates([make_review("same text here", review_id="a"),
                                  make_review("same text here", review_id="b")])
        assert [r.is_duplicate for r in marked] == [False, True]


class TestQuality:
    def test_short_text_scores_low(self):
        assert quality_weight("good") <= 0.2

    def test_detailed_problem_report_scores_high(self):
        text = ("The app crashes every time I try to submit an order and I have to "
                "reinstall it, so I use the website instead now")
        assert quality_weight(text) > 0.7

    def test_repetitive_spam_penalised_by_diversity(self):
        assert lexical_diversity("good good good good") < 0.3

    def test_pure_praise_detected(self):
        assert is_pure_praise("great app love it")

    def test_praise_with_a_complaint_is_not_pure(self):
        assert not is_pure_praise("great app but it crashes on login")

    def test_empty_text_weight_is_zero(self):
        assert quality_weight("") == 0.0


class TestPreprocessPipeline:
    def test_end_to_end(self, reviews):
        result = ReviewPreprocessor().run(reviews)
        assert result.diagnostics.n_input == len(reviews)
        assert result.clusterable_units
        assert result.diagnostics.n_duplicates >= 1

    def test_praise_excluded_from_clustering_but_review_kept(self, reviews):
        result = ReviewPreprocessor().run(reviews)
        assert "praise1" in result.reviews_by_id()
        assert not any(u.review_id == "praise1" for u in result.clusterable_units)

    def test_duplicates_excluded_from_clustering(self, reviews):
        result = ReviewPreprocessor().run(reviews)
        assert not any(u.review_id == "dupe1" for u in result.clusterable_units)

    def test_non_english_dropped_when_configured(self):
        reviews = [make_review("アプリがクラッシュします", review_id="jp"),
                   make_review("the app crashes when I open my order history", review_id="en")]
        result = ReviewPreprocessor(PreprocessConfig(drop_non_language=True)).run(reviews)
        assert result.diagnostics.n_wrong_language == 1

    def test_empty_input_is_safe(self):
        result = ReviewPreprocessor().run([])
        assert result.clusterable_units == [] and result.diagnostics.n_input == 0
