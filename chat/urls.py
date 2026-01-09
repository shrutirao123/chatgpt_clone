from django.urls import path
from . import views # Import all views

urlpatterns = [
    # Authentication
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),

    # Chat UI Views
    # 1. Base URL (redirects to the latest/newest conversation)
    path('', views.chat_ui, name='chat_ui_base'), 
    # 2. Conversation-specific URL (for page switching)
    path('conversation/<int:conversation_id>/', views.chat_ui, name='chat_ui_conversation'),

    # API Endpoints
    path('api/chat_with_ai/', views.chat_with_ai, name='chat_with_ai'),
    path('api/new_conversation/', views.new_conversation, name='new_conversation'),
    path("api/delete_conversation/<int:conv_id>/", views.delete_conversation, name="delete_conversation"),
    path("api/stream_chat_with_ai/", views.stream_chat_with_ai, name="stream_chat_with_ai"),
    path('api/rename_conversation/<int:conv_id>/', views.rename_conversation, name='rename_conversation'),
    path('api/toggle_archive_conversation/<int:conv_id>/', views.toggle_archive_conversation, name='toggle_archive_conversation'),
]