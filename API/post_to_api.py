from API_modules import post_to_our_API


if __name__ == "__main__":

    text = "The Chairman of the Republican Party of Texas said the recovery rate for COVID-19 is 99.9% in Texas. That’s False. @PolitiFactTexas https://t.co/GmUXoVT2Dh https://t.co/ltIHWPchJM"
    first_post = post_to_our_API(text, model="logistic")


    print(first_post)