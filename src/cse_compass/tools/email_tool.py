import os

import sendgrid
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sendgrid.helpers.mail import Content, Email, Mail, To

load_dotenv()


class EmailInput(BaseModel):
    """Input for sending email."""

    body: str = Field(description="Email body content")


def send_email(body: str):
    """Send out an email with the given body to users."""
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get("SENDGRID_API_KEY"))
    from_email = Email("prabhash@beatzlane.com")
    to_email = To("prabhashdilhanakmeemana@gmail.com")
    content = Content("text/plain", body)
    mail = Mail(from_email, to_email, "CSE Compass Recommendation Report", content).get()
    sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}


def get_email_tool():
    return StructuredTool.from_function(
        name="email",
        description="Send an email to a recipient",
        func=send_email,
        args_schema=EmailInput,
    )