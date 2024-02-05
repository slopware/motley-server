# Download the helper library from https://www.twilio.com/docs/python/install
import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

# Set environment variables for your credentials
# Read more at http://twil.io/secure


account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

call = client.calls.create(
  url="http://demo.twilio.com/docs/voice.xml",
  to=os.environ["MY_NUMBER"],
  from_=os.environ["TWILIO_NUMBER"]
)

print(call.sid)