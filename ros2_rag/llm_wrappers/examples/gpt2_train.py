
from datasets import load_dataset

from gpt2_wrapper.gpt2_wrapper import GPT2Type, GPT2Wrapper


def main():
    # Create wrapper
    gpt2_wrapper = GPT2Wrapper(GPT2Type.BASE)
    # Load pre-trained
    gpt2_wrapper.load_pretrained_model()
    # Generate
    input = "Q: If train A moves with a speed of 50Km/h, which is the speed of train A in m/s?\nA:"
    res, output = gpt2_wrapper.generate(input)
    print("INPUT: " + input)
    print("OUTPUT: " + output)

    # Load training dataset
    print("Loading dataset 'dim/grade_school_math_instructions_3k'...")
    dataset = load_dataset("dim/grade_school_math_instructions_3k")

    gpt2_wrapper.train_model(dataset, "INSTRUCTION", "RESPONSE",
                             "./trained_gpt2_school_math",
                             5, 0.00025, True)

    # Load trained model
    trained_gpt2_wrapper = GPT2Wrapper()
    trained_gpt2_wrapper.load_stored_model("./trained_gpt2_school_math")
    # Generate
    res, trained_output = trained_gpt2_wrapper.generate(input)
    print("TRAINED OUTPUT: " + trained_output)


if __name__ == "__main__":
    main()
