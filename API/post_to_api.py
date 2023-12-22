from API_modules import post_to_our_API,get_csrf_token


if __name__ == "__main__":
    # key = get_csrf_token()
    text = "The Chairman of the Republican Party of Texas said the recovery rate for COVID-19 is 99.9% in Texas. That’s False. @PolitiFactTexas https://t.co/GmUXoVT2Dh https://t.co/ltIHWPchJM"
    first_post = post_to_our_API(text, model="svm")


    print(first_post)