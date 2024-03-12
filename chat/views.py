from django.shortcuts import render
from django.http import JsonResponse

from django.utils import timezone
from django.contrib.auth.decorators import login_required

from .models import Chat, Document
from .chat import ask, LLM, EMBEDDING_MODEL
from .rag import DocumentPDF


@login_required(login_url="/accounts/login")
def chat_service(request):
    chats = Chat.objects.filter(user=request.user)

    if request.method == "POST":
        if request.POST.get("chat", ""):
            return chat(request)
        if request.POST.get("refresh", ""):
            return refresh_chat(request)
        if request.POST.get("upload", ""):
            return upload_document(request)

    return render(request, "chat.html", {"chats": chats})


@login_required(login_url="/accounts/login")
def chat(request):
    message = request.POST.get("message")
    response = ask(message, LLM, EMBEDDING_MODEL)

    chat = Chat(
        user=request.user,
        message=message,
        response=response,
        created_at=timezone.now(),
    )
    chat.save()
    return JsonResponse({"message": message, "response": response})


@login_required(login_url="/accounts/login")
def refresh_chat(request):
    Chat.objects.filter(user=request.user).delete()
    return JsonResponse({"status": "success", "message": "Chat history cleared."})


@login_required(login_url="/accounts/login")
def upload_document(request):
    document = Document(user=request.user, document=request.FILES["document"])
    document.save()
    DocumentPDF(document.document.path).embedding()
    return JsonResponse({"message": "Document uploaded successfully"})
