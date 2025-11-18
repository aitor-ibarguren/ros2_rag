
from datasets import load_dataset

from gpt2_wrapper.gpt2_wrapper import GPT2Wrapper


def main():
    # Create wrapper
    gpt2_wrapper = GPT2Wrapper()
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

    gpt2_wrapper.peft_lora_train_model(dataset, "INSTRUCTION", "RESPONSE",
                                       "./peft_trained_gpt2_school_math",
                                       0.25, 1e-4, True)

    # Load trained model
    peft_trained_gpt2_wrapper = GPT2Wrapper()
    peft_trained_gpt2_wrapper.load_stored_peft_model(
        "./peft_trained_gpt2_school_math"
    )
    # Generate
    res, peft_trained_output = peft_trained_gpt2_wrapper.generate(input)
    print("PEFT TRAINED OUTPUT: " + peft_trained_output)


if __name__ == "__main__":
    main()
