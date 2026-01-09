from django.contrib import admin
from .models import Conversation, ChatHistory

# --- Conversation Admin ---

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'user', 'created_at')
    # Allows filtering the list view by user
    list_filter = ('user', 'created_at')
    # Allows searching by title or username
    search_fields = ('title', 'user__username')
    # Makes the user field read-only after creation
    readonly_fields = ('user', 'created_at')


# --- ChatHistory Admin ---

@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'conversation', 'sender', 'content_snippet', 'created_at')
    # Allows filtering the list view by sender and conversation
    list_filter = ('sender', 'conversation__user', 'created_at')
    # Allows searching the content
    search_fields = ('content',)
    # Makes content and conversation read-only
    readonly_fields = ('conversation', 'sender', 'created_at')

    def content_snippet(self, obj):
        """Display a truncated version of the content in the list view."""
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_snippet.short_description = 'Content'
