from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Selecting the model. You will be using "facebook/blenderbot-400M-distill" in this example.
model_name = "facebook/blenderbot-400M-distill"
model_name_two = "google/flan-t5-base"

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_two)
tokenizer = AutoTokenizer.from_pretrained(model_name_two)


def chat_with_bot():
    while True:
        # Get user input
        input_text = input("You: ")

        # Exit conditions
        if input_text.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Tokenize input and generate response
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Display bot's response
        print("Chatbot:", response)


def chat_with_another_bot():
    while True:
        # Get user input
        input_text = input("You: ")

        # Exit conditions
        if input_text.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Tokenize input and generate response
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Display bot's response
        print("Chatbot:", response)


chat_with_another_bot()
