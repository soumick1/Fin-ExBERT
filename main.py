#from models import *
#from preprocess_data import *
from utils import *
from time import time
import logging


if __name__ == '__main__':
    # train_model_with_chkpt(epochs=5, batch_size=16, lr=2e-3,
    #             save_model=True,
    #             save_path='gnn_model_checkpoint.pt',
    #             resume=True)

    sample_transcript = """
    Agent: Hello, thank you for calling Acme Financial Services. My name is Priya. How can I help you today?
    Customer: Hi Priya, I’m considering opening a new savings account with you.
    Agent: Absolutely—our savings account offers 4% interest per annum. Do you have a balance in mind?
    Customer: Yes, I’d like to deposit ₹50,000 initially, and then I’m interested in investing another ₹2 lakh in mutual funds over the next month.
    Agent: Great, we have several mutual fund options. Are you more growth-oriented or looking for steady income?
    Customer: I want to focus on growth. Also, could you tell me about your home loan rates? I may need a ₹30 lakh mortgage in the next six months.
    Agent: Certainly—we currently offer home loan rates starting at 6.8%. Do you already own property or are you planning to buy?
    Customer: Planning to buy. Finally, I’d like to apply for a credit card with a high cashback—maybe one that gives 2% on all spends.
    Agent: We have a Platinum Cashback Card at 1.5%, and our Signature Cashback Card at 2%. Would you like me to initiate the application?
    Customer: Yes please, go ahead with the Signature Cashback Card, and send me the home-loan documents via email.
    Agent: Done. You’ll receive an email shortly. Is there anything else I can help you with?
    Customer: No, that’s all for today—thank you!
    """

    complex_transcript = """
    Agent: Good morning, thank you for calling Maple Grove Bank. This is Rahul speaking—how may I assist you today?
    Customer: Hi Rahul, I’ve been reviewing my financial goals for the next five years and want to discuss a mix of savings, investments, and insurance.
    Agent: Absolutely. Would you like to start with your current cash savings or jump straight into investment products?
    Customer: Let’s begin with savings: I’d like to open a high-yield savings account with at least ₹1 lakh to start, and then set up an automatic top-up of ₹10,000 each month.
    Agent: Great choice. We have our “Plus Savings” account at 4.2% APY. Next, investments—are you looking at mutual funds, stocks, or retirement plans?
    Customer: I’m particularly interested in tax-saving ELSS mutual funds and a more conservative retirement pension plan. Also, could you explain your term insurance offerings?
    Agent: Sure—our ELSS options include Fund A (equity-heavy) and Fund B (balanced). For term cover, we have 20-year plans up to ₹50 lakhs. Any preference?
    Customer: I want a balanced ELSS with a 3-year lock-in, and term insurance of ₹30 lakhs for 25 years. After that, I may need advice on buying a second home—so let’s also discuss mortgage pre-approval.
    Agent: Understood. For a ₹30 lakh home loan, current interest rates start at 6.9%. We can pre-approve you based on your income. Shall I proceed?
    Customer: Yes, please initiate the home-loan pre-approval. And lastly, I’d like to apply for a debit card with no annual fee and a co-branded credit card offering travel rewards.
    Agent: Certainly—our “Freedom” debit card has no fee, and the “SkyMiles” credit card gives 2 airline miles per ₹100 spent. Would you like to complete those applications now?
    Customer: Yes, go ahead with both. Also, can you set up a quarterly portfolio review call with a financial advisor?
    Agent: Absolutely. I’ll schedule a review every three months starting next quarter. You’ll get email confirmations shortly.
    Customer: Perfect—that covers all my needs. Thanks for your help!
    Agent: My pleasure! Have a great day and feel free to call back anytime.
    """

    # premise_input = "personA is on the stage giving a speech."
    # hypothesis_input = "personA is using a microphone."
    # prediction, _ = predict_fin_nli(premise=premise_input, hypothesis=hypothesis_input, model_path='gnn_model_checkpoint.pt')
    # print("Prediction:", prediction)
    # print('Final layer logits:', _)

    start = time()
    results = extract_sentences_by_intent(
        complex_transcript,
        intent="customer tells about own financial condition",#"customer states specific financial product requests and planning preferences",
        #"agent provides assistance", #"customer states their financial needs",
        threshold=0.60,
        top_k=10,
        convo_focus='customer'
    )
    end = time()

    logging.info('Prediction Done in {:.2f}sec'.format(end - start))

    for sentence, score in results:
        print(f"{score:.2f} → {sentence}")

