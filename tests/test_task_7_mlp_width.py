import unittest

from task_7 import task_7


class MLPWidthTests(unittest.TestCase):
    def test_model_factory_builds_requested_mlp_width(self):
        self.assertTrue(hasattr(task_7, "build_model"), "build_model is not implemented")

        model = task_7.build_model("mlp", side=32, n_classes=7, mlp_width=256)

        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 296967)

    def test_model_factory_preserves_existing_default_size(self):
        self.assertTrue(hasattr(task_7, "build_model"), "build_model is not implemented")

        model = task_7.build_model("mlp", side=32, n_classes=7, mlp_width=512)

        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 659463)

    def test_cnn_checkpoint_uses_default_width_placeholder(self):
        self.assertTrue(
            hasattr(task_7, "checkpoint_mlp_width"),
            "checkpoint_mlp_width is not implemented",
        )

        width = task_7.checkpoint_mlp_width({"model_kind": "cnn", "mlp_width": None})

        self.assertEqual(width, 512)

    def test_checkpoint_rejects_fractional_mlp_width(self):
        with self.assertRaisesRegex(ValueError, "mlp_width"):
            task_7.checkpoint_mlp_width({"model_kind": "mlp", "mlp_width": 256.5})


if __name__ == "__main__":
    unittest.main()
