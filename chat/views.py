# chatapp/views.py
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .rag import medical_chatbot_modern

@csrf_exempt
def medico(request):
    ai_response = ""
    
    if request.method == "POST":
        user_input = request.POST.get("question", "")
        
        if user_input:
            ai_response = medical_chatbot_modern(user_input)
    
    return render(request, 'medico.html', {"response": ai_response})

# Create your views here.
def chat(request):
    return render(request,'chat.html')
