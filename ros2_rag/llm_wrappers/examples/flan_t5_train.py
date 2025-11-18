
from datasets import load_dataset

from flan_t5_wrapper.flan_t5_wrapper import FlanT5Type, FlanT5Wrapper


def main():
    # Create wrapper
    flan_t5_wrapper = FlanT5Wrapper(FlanT5Type.SMALL)
    # Load pre-trained
    flan_t5_wrapper.load_pretrained_model()
    # Generate
    input = "If train A moves with a speed of 50Km/h, which is the speed of train A in m/s?"
    res, output = flan_t5_wrapper.generate(input)
    print("INPUT: " + input)
    print("OUTPUT: " + output)

    # Load training dataset
    print("Loading dataset 'dim/grade_school_math_instructions_3k'...")
    dataset = load_dataset("dim/grade_school_math_instructions_3k")

    flan_t5_wrapper.train_model(dataset, "INSTRUCTION", "RESPONSE",
                                "./trained_flan_t5_school_math",
                                5, 0.00025, True)

    # Load trained model
    trained_flan_t5_wrapper = FlanT5Wrapper()
    trained_flan_t5_wrapper.load_stored_model("./trained_flan_t5_school_math")
    # Generate
    res, trained_output = trained_flan_t5_wrapper.generate(input)
    print("TRAINED OUTPUT: " + trained_output)


if __name__ == "__main__":
    main()
