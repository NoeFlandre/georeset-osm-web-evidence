import unittest

from georeset_osm_web_evidence.osm.worldwide_bboxes import (
    BASE_WORLDWIDE_TRAINING_BBOX_COUNT,
    WORLDWIDE_PILOT_BBOXES,
    WORLDWIDE_TRAINING_BBOXES,
    generate_bbox_expansions,
)


class WorldwideBboxTests(unittest.TestCase):
    def test_worldwide_catalog_includes_base_and_expanded_bboxes(self) -> None:
        self.assertGreater(
            len(WORLDWIDE_TRAINING_BBOXES),
            BASE_WORLDWIDE_TRAINING_BBOX_COUNT,
        )
        self.assertEqual(WORLDWIDE_PILOT_BBOXES, WORLDWIDE_TRAINING_BBOXES[:16])
        self.assertTrue(
            any(
                "_exp_" in bbox["bbox_id"]
                for bbox in WORLDWIDE_TRAINING_BBOXES[
                    BASE_WORLDWIDE_TRAINING_BBOX_COUNT:
                ]
            )
        )

    def test_generate_bbox_expansions_preserves_anchor_metadata(self) -> None:
        anchor = {
            "bbox_id": "fr_test",
            "bbox_label": "France test",
            "country": "France",
            "world_region": "Europe",
            "local_language": "fr",
            "bbox": (47.8, -4.4, 48.2, -3.7),
        }

        expansions = generate_bbox_expansions([anchor], expansion_radius_steps=1)

        self.assertEqual(len(expansions), 8)
        self.assertEqual({bbox["country"] for bbox in expansions}, {"France"})
        self.assertEqual({bbox["world_region"] for bbox in expansions}, {"Europe"})


if __name__ == "__main__":
    unittest.main()
