from django.db import models
from django.contrib.auth.models import User

class Conversation(models.Model):
    """Represents a single chat session for a user."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="conversations")
    # Title will be updated with the first user message
    title = models.CharField(max_length=255, default="New Chat") 
    created_at = models.DateTimeField(auto_now_add=True)
    is_archived = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.title} ({self.user.username})"


class ChatHistory(models.Model):
    """Represents an individual message (user or bot) within a conversation."""
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="messages",
        # NOTE: null/blank removed for data integrity. Every message must have a conversation.
    )
    sender = models.CharField(
        max_length=10,
        choices=[("user", "User"), ("bot", "Bot"), ("system", "System")],  # Added "system" for file notifications
        default="user" 
    )
    # Storing the content of the message
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender}: {self.content[:30]}..."