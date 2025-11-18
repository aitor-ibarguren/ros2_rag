import os
import sys
import unittest

if __name__ == "__main__" or __package__ is None:
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )

from datasets import load_dataset

from flan_t5_wrapper.flan_t5_wrapper import FlanT5Type, FlanT5Wrapper


class TestFlanT5Wrapper(unittest.TestCase):
    def test_load_pretrained_model(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper(model_type=FlanT5Type.BASE)
        # Check if pretrained model loads
        self.assertTrue(flan_t5_wrapper.load_pretrained_model())
        # Check if multiple-loading detected
        self.assertTrue(not flan_t5_wrapper.load_pretrained_model())

    def test_save_and_load_model(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Check if saves when no model is available
        self.assertFalse(flan_t5_wrapper.save_model("./saved_model"))
        # Check if pretrained model loads
        self.assertTrue(flan_t5_wrapper.load_pretrained_model())
        # Check if saves model & tokenizer
        self.assertTrue(flan_t5_wrapper.save_model("./saved_model"))
        # Clear model
        self.assertTrue(flan_t5_wrapper.clear_model())
        # Try to load model from folder
        self.assertTrue(flan_t5_wrapper.load_stored_model("./saved_model"))
        # Try to load model from folder againAdd commentMore actions
        self.assertFalse(flan_t5_wrapper.load_stored_model("./saved_model"))

    def test_load_model_2(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Try to load non-existent folder
        self.assertFalse(flan_t5_wrapper.load_stored_model("./no-model"))
        self.assertFalse(flan_t5_wrapper.load_stored_peft_model("./no-model"))

    def test_generate(self):
        # Prepare inputs
        input = "The result of this Python unit test will be successful or not?"
        inputs = [
            "Answer the following question: "
            "Is Paris in France?",
            "Answer the following question: "
            "Is Python a programming language?"
        ]
        # Create wrapperAdd commentMore actions
        flan_t5_wrapper = FlanT5Wrapper()
        # Generate without model
        res, output = flan_t5_wrapper.generate(input)
        self.assertFalse(res)
        # Generate list without model
        res, outputs = flan_t5_wrapper.generate_list(inputs)
        self.assertFalse(res)
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Check if pretrained model loads
        self.assertTrue(flan_t5_wrapper.load_pretrained_model())
        # Generate output
        res, output = flan_t5_wrapper.generate(input)
        print("INPUT: " + input)
        print("OUTPUT: " + output)
        self.assertTrue(res and len(output) > 0)
        # Generate output from list
        res, outputs = flan_t5_wrapper.generate_list(inputs)
        for input_str, output_str in zip(inputs, outputs):
            print("*******")
            print("INPUTS: " + input_str)
            print("OUTPUTS: " + output_str)
        self.assertTrue(res and len(outputs) == 2)

    def test_train_model(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Load pre-trained
        flan_t5_wrapper.load_pretrained_model()
        # Load training dataset
        print("Loading dataset 'dim/grade_school_math_instructions_3k'...")
        dataset = load_dataset("dim/grade_school_math_instructions_3k")
        # Take few for testing
        dataset = dataset.filter(lambda example, index: index % 100 == 0,
                                 with_indices=True)
        # Train model
        save_folder = "./trained_flan_t5_school_math"
        res = flan_t5_wrapper.train_model(dataset, "INSTRUCTION", "RESPONSE",
                                          save_folder, 0.05, 0.00025, True)
        self.assertTrue(res)
        # Load trained model
        trained_flan_t5_wrapper = FlanT5Wrapper()
        self.assertTrue(trained_flan_t5_wrapper.load_stored_model(save_folder))

    def test_peft_train_model(self):
        # Create wrapper
        flan_t5_wrapper = FlanT5Wrapper()
        # Load pre-trained
        flan_t5_wrapper.load_pretrained_model()
        # Load training dataset
        print("Loading dataset 'dim/grade_school_math_instructions_3k'...")
        dataset = load_dataset("dim/grade_school_math_instructions_3k")
        # Take few for testing
        dataset = dataset.filter(lambda example, index: index % 100 == 0,
                                 with_indices=True)
        # Train PEFT model
        save_folder = "./peft_trained_flan_t5_school_math"
        res = flan_t5_wrapper.peft_lora_train_model(dataset, "INSTRUCTION",
                                                    "RESPONSE", save_folder,
                                                    0.05, 1e-4, True)
        self.assertTrue(res)
        # Load trained model
        peft_trained_flan_t5_wrapper = FlanT5Wrapper()
        self.assertTrue(peft_trained_flan_t5_wrapper.load_stored_peft_model(
            save_folder
        ))
        # Try to load model again
        self.assertFalse(peft_trained_flan_t5_wrapper.load_stored_peft_model(
            save_folder
        ))


if __name__ == "__main__":
    unittest.main()
