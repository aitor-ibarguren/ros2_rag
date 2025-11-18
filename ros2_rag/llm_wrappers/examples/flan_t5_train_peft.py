
from datasets import load_dataset

from flan_t5_wrapper.flan_t5_wrapper import FlanT5Wrapper


def main():
    # Create wrapper
    flan_t5_wrapper = FlanT5Wrapper()
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

    flan_t5_wrapper.peft_lora_train_model(dataset, "INSTRUCTION", "RESPONSE",
                                          "./peft_trained_flan_t5_school_math",
                                          10.0, 1e-4, True)

    # Load trained model
    peft_trained_flan_t5_wrapper = FlanT5Wrapper()
    peft_trained_flan_t5_wrapper.load_stored_peft_model(
        "./peft_trained_flan_t5_school_math"
    )
    # Generate
    res, peft_trained_output = peft_trained_flan_t5_wrapper.generate(input)
    print("PEFT TRAINED OUTPUT: " + peft_trained_output)


if __name__ == "__main__":
    main()
