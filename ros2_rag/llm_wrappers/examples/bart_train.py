
from datasets import load_dataset

from bart_wrapper.bart_wrapper import BARTType, BARTWrapper


def main():
    # Create wrapper
    bart_wrapper = BARTWrapper(BARTType.BASE)
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

    bart_wrapper.train_model(dataset, "INSTRUCTION", "RESPONSE",
                             "./trained_bart_school_math",
                             5.0, 0.00025, True)

    # Load trained model
    trained_bart_wrapper = BARTWrapper()
    trained_bart_wrapper.load_stored_model("./trained_bart_school_math")
    # Generate
    res, trained_output = trained_bart_wrapper.generate(input)
    print("TRAINED OUTPUT: " + trained_output)


if __name__ == "__main__":
    main()
