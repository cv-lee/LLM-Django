from django.db import models
from django.contrib.auth.models import User


class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}: {self.message}"


class Document(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    document = models.FileField(upload_to="chat/chroma_db/documents")
    is_embedding = models.BooleanField(default=False)
    is_public = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}: {self.document.name}"
