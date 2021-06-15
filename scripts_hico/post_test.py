import requests

def send_message_to_slack(text):
    url = "https://hooks.slack.com/services/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"

    payload = {"text" : text}

    requests.post(url, json=payload)

def send_message_to_slack_baseline(text):
    url = "https://hooks.slack.com/services/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"

    payload = {"text" : text}

    requests.post(url, json=payload)

