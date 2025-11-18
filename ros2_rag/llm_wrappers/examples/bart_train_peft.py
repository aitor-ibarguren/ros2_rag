
from datasets import load_dataset

from bart_wrapper.bart_wrapper import BARTWrapper


def main():
    # Create wrapper
    bart_wrapper = BARTWrapper()
    # Load pre-trained
    bart_wrapper.load_pretrained_model()
    # Generate
    input = "If train A moves with a speed of 50Km/h, which is the speed of train A in m/s?"
    res, output = bart_wrapper.generate(input)
    print("INPUT: " + input)
    print("OUTPUT: " + output)

    # Load training dataset
    print("Loading dataset 'dim/grade_school_math_instructions_3k'...")
    dataset = load_dataset("dim/grade_school_math_instructions_3k")

    bart_wrapper.peft_lora_train_model(dataset, "INSTRUCTION", "RESPONSE",
                                       "./peft_trained_bart_school_math",
                                       10.0, 1e-4, True)

    # Load trained model
    peft_trained_bart_wrapper = BARTWrapper()
    peft_trained_bart_wrapper.load_stored_peft_model(
        "./peft_trained_bart_school_math"
    )
    # Generate
    res, peft_trained_output = peft_trained_bart_wrapper.generate(input)
    print("PEFT TRAINED OUTPUT: " + peft_trained_output)


if __name__ == "__main__":
    main()
