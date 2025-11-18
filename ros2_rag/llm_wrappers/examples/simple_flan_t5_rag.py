import os

from faiss_wrapper.faiss_wrapper import FAISSWrapper
from flan_t5_wrapper.flan_t5_wrapper import FlanT5Type, FlanT5Wrapper


def main():
    # Create Flan T5 wrapper
    generator = FlanT5Wrapper(FlanT5Type.SMALL)
    # Load pre-trained
    generator.load_pretrained_model()

    # Create retriever wrapper
    retriever = FAISSWrapper()
    # Init new index
    retriever.init_new_index()
    # Add CSV to index
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'data', 'shop_data.csv')
    retriever.add_from_csv(file_path, 'shop data')

    # Prompt
    prompts = ['Can I buy a Gibson guitar at your shop?',
               'Which is the return policy?']

    # Get relevant data
    res, relevant_data, _ = retriever.search(prompts, 5)

    if not res:
        print('Could not retrieve relevant information for the prompts')
        return

    # Generate augmented prompt
    augmented_prompts = []

    for index, relevant_text_list in enumerate(relevant_data):
        # Set all relevan text in a string
        context = ''
        for text in relevant_text_list:
            context += text + '\n'

        augmented_prompts.append(
            f"You are a helpful assistant answering customer questions.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{prompts[index]}\n\n"
            f"Answer the question using only the information in the context"
            f"above. Be concise and accurate."
        )

    # Generate
    res, outputs = generator.generate_list(augmented_prompts)
    for input_str, output_str in zip(prompts, outputs):
        print('INPUT: ' + input_str)
        print('OUTPUT: ' + output_str)


if __name__ == '__main__':
    main()
