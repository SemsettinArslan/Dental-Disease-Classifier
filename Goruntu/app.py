import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dental disease description mapping
disease_info = {
    "Calculus": {
        "title": "Diş Taşı (Calculus)",
        "description": "Plak birikiminin mineralize olup sertleşmesiyle oluşan tartar yapısıdır. Diş eti çizgisi yakınlarında sarı/kahverengi lekelenmeler şeklinde görülür ve evde fırçalama ile temizlenemez.",
        "recommendation": "Diş taşı temizliği (detertraj) ve profesyonel diş bakımı için diş hekiminizi ziyaret etmelisiniz. Temizlenmeyen diş taşları ilerleyen dönemde diş eti çekilmesine ve periodontal hastalıklara yol açabilir."
    },
    "Caries": {
        "title": "Diş Çürüğü (Caries)",
        "description": "Ağız içindeki bakterilerin gıda kalıntılarıyla beslenerek asit üretmesi ve bu asitlerin diş minesini aşındırarak çürüklere sebep olması durumudur.",
        "recommendation": "Çürüğün ilerlemesini durdurmak ve sinir dokusuna (pulpa) ulaşmasını engellemek için en kısa sürede diş dolgusu yaptırmalısınız. Erken tedavi, kanal tedavisi ve diş kaybını önler."
    },
    "Gingivitis": {
        "title": "Diş Eti İltihabı (Gingivitis)",
        "description": "Diş etlerinin kızarık, şiş ve fırçalama esnasında kolayca kanar hale gelmesiyle karakterize edilen, en sık görülen ve erken evre diş eti hastalığıdır.",
        "recommendation": "Günde en az iki kez doğru teknikle dişlerinizi fırçalamalı, her gün diş ipi veya arayüz fırçası kullanmalısınız. Diş hekiminizden diş taşı temizliği randevusu alarak diş eti sağlığınızı koruyabilirsiniz."
    },
    "Hypodontia": {
        "title": "Doğuştan Diş Eksikliği (Hypodontia)",
        "description": "Bir veya birden fazla dişin gelişimsel olarak çenede hiç oluşmaması durumudur. Genellikle genetik faktörlere bağlıdır ve çiğneme düzenini, konuşmayı veya çene kemiğini etkileyebilir.",
        "recommendation": "Eksikliğin konumuna ve yaşınıza göre implant, diş köprüsü veya ortodontik (tel) tedaviler ile boşluğun kapatılması gerekebilir. Diş hekiminizle detaylı bir tedavi planı oluşturmalısınız."
    },
    "Mouth Ulcer": {
        "title": "Ağız Yarası (Mouth Ulcer)",
        "description": "Ağız içinde, dil, yanak veya diş etlerinde oluşan, kenarları kırmızı, ortası beyaz/gri renkli, oldukça ağrılı açık lezyonlardır. Stres, vitaminsizlik veya ağız içi travmalar tetikleyebilir.",
        "recommendation": "Aftlar genellikle 7-10 gün içinde kendiliğinden geçer. Bu süreçte asitli, tuzlu, çok sıcak veya acı gıdalardan kaçınmalısınız. Eczacınıza danışarak ağrıyı azaltan ve iyileşmeyi hızlandıran jeller kullanabilirsiniz."
    },
    "Tooth Discoloration": {
        "title": "Diş Renklenmesi (Tooth Discoloration)",
        "description": "Dişlerin dış yüzeyinde çay, kahve, sigara tüketimi gibi dışsal nedenlerle veya florozis, ilaç kullanımı gibi içsel nedenlerle oluşan sararma veya grileşmelerdir.",
        "recommendation": "Yüzeysel lekelenmeler için profesyonel diş temizliği veya beyazlatıcı diş macunları yardımcı olabilir. Daha kalıcı ve estetik bir çözüm için klinik ortamda diş beyazlatma (bleaching) yaptırabilirsiniz."
    }
}

# Global variables for model
model = None
device = None
class_names = []
val_transforms = None

def load_model():
    global model, device, class_names, val_transforms
    model_path = "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found! Run training first. App will start in placeholder mode.")
        return False
        
    print(f"Loading weights from {model_path} on {device}...")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    
    # Initialize architecture
    model = resnet34()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("Model loaded successfully!")
    return True

import json

@app.route('/')
def index():
    metrics_data = None
    metrics_path = "metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception as e:
            print(f"Error loading metrics.json: {e}")
    return render_template('index.html', metrics=metrics_data)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model eğitilmemiş veya yüklenmemiş. Lütfen önce modeli eğitin."}), 500

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Dosya yüklenmedi."}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "Geçersiz dosya ismi."}), 400
        
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process image
            image = Image.open(file_path).convert("RGB")
            input_tensor = val_transforms(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                
            conf, pred_class_idx = torch.max(probabilities, 0)
            pred_class = class_names[pred_class_idx.item()]
            confidence = conf.item()
            
            # Map predictions
            all_predictions = []
            for idx, prob in enumerate(probabilities):
                all_predictions.append({
                    "class_name": class_names[idx],
                    "display_name": disease_info.get(class_names[idx], {}).get("title", class_names[idx]),
                    "probability": float(prob.item())
                })
            
            # Sort by probability descending
            all_predictions = sorted(all_predictions, key=lambda x: x['probability'], reverse=True)
            
            info = disease_info.get(pred_class, {
                "title": pred_class,
                "description": "Bilinmeyen sınıf tanımı.",
                "recommendation": "Lütfen diş hekiminize danışın."
            })
            
            # Clean up uploaded file
            os.remove(file_path)
            
            return jsonify({
                "success": True,
                "prediction": pred_class,
                "display_name": info["title"],
                "confidence": confidence,
                "description": info["description"],
                "recommendation": info["recommendation"],
                "all_predictions": all_predictions
            })
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"success": False, "error": f"Görsel işlenirken hata oluştu: {str(e)}"}), 500

@app.route('/plot/<filename>')
def get_plot(filename):
    if filename in ['learning_curves.png', 'confusion_matrix.png']:
        return send_file(os.path.join(os.path.dirname(__file__), filename))
    return "Not Found", 404

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
