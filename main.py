#from models import *
#from preprocess_data import *
from utils import *
from time import time
import logging


if __name__ == '__main__':
    train_model_with_chkpt(epochs=6, batch_size=16, lr=2e-5,
                save_model=True,
                save_path='gnn_model_checkpoint.pt',
                resume=True)

    # sample_transcript = """
    # Agent: Hello, thank you for calling Acme Financial Services. My name is Priya. How can I help you today?
    # Customer: Hi Priya, I’m considering opening a new savings account with you.
    # Agent: Absolutely—our savings account offers 4% interest per annum. Do you have a balance in mind?
    # Customer: Yes, I’d like to deposit ₹50,000 initially, and then I’m interested in investing another ₹2 lakh in mutual funds over the next month.
    # Agent: Great, we have several mutual fund options. Are you more growth-oriented or looking for steady income?
    # Customer: I want to focus on growth. Also, could you tell me about your home loan rates? I may need a ₹30 lakh mortgage in the next six months.
    # Agent: Certainly—we currently offer home loan rates starting at 6.8%. Do you already own property or are you planning to buy?
    # Customer: Planning to buy. Finally, I’d like to apply for a credit card with a high cashback—maybe one that gives 2% on all spends.
    # Agent: We have a Platinum Cashback Card at 1.5%, and our Signature Cashback Card at 2%. Would you like me to initiate the application?
    # Customer: Yes please, go ahead with the Signature Cashback Card, and send me the home-loan documents via email.
    # Agent: Done. You’ll receive an email shortly. Is there anything else I can help you with?
    # Customer: No, that’s all for today—thank you!
    # """

    # premise_input = "personA is on the stage giving a speech."
    # hypothesis_input = "personA is using a microphone."
    # prediction, _ = predict_fin_nli(premise=premise_input, hypothesis=hypothesis_input, model_path='gnn_model_checkpoint.pt')
    # print("Prediction:", prediction)
    # print('Final layer logits:', _)

    # start = time()
    # results = extract_sentences_by_intent(
    #     sample_transcript,
    #     intent="agent provides assistance", #"customer states their financial needs",
    #     threshold=0.8,
    #     top_k=5,
    #     convo_focus='agent'
    # )
    # end = time()
    #
    # logging.info('Prediction Done in {:.2f}sec'.format(end - start))
    #
    # for sentence, score in results:
    #     print(f"{score:.2f} → {sentence}")

