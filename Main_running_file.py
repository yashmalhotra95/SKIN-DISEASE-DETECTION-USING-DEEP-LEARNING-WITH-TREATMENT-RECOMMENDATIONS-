import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from keras.models import load_model
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2

class SkinDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Skin Disease Detection System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg="#f5f5f5")
        
        # Load model and labels
        self.model = load_model("keras_Model.h5", compile=False)
        self.class_names = [line.strip() for line in open("labels.txt", "r").readlines()]
        
        # Medical recommendations database
        self.MEDICAL_RECOMMENDATIONS = {
    "Acne": {
        "medications": [
            "Benzoyl peroxide (2.5-10%)",
            "Salicylic acid (0.5-2%)",
            "Topical retinoids (Tretinoin 0.025-0.1%)",
            "Topical antibiotics (Clindamycin 1%)",
            "Oral antibiotics (Doxycycline, Minocycline)",
            "Oral contraceptives (for hormonal acne)",
            "Isotretinoin (for severe cystic acne)"
        ],
        "precautions": [
            "Wash face twice daily with gentle cleanser",
            "Avoid picking or squeezing lesions",
            "Use non-comedogenic skincare products",
            "Change pillowcases frequently",
            "Reduce stress and maintain healthy diet",
            "Avoid excessive sun exposure",
            "Gradually introduce new treatments to avoid irritation"
        ]
    },
    "Actinic Keratosis/Basal Cell Carcinoma": {
        "medications": [
            "Fluorouracil cream (5%)",
            "Imiquimod cream (5%)",
            "Diclofenac gel (3%)",
            "Ingenol mebutate gel",
            "Cryotherapy with liquid nitrogen",
            "Photodynamic therapy",
            "Surgical excision for confirmed BCC"
        ],
        "precautions": [
            "Immediate dermatologist consultation required",
            "Regular skin cancer screenings every 6-12 months",
            "Daily broad-spectrum SPF 30+ sunscreen",
            "Protective clothing and wide-brimmed hats",
            "Avoid peak sun hours (10am-4pm)",
            "Avoid tanning beds",
            "Monitor for changes in size, color, or texture"
        ]
    },
    "Atopic Dermatitis": {
        "medications": [
            "Topical corticosteroids (Hydrocortisone 1%)",
            "Calcineurin inhibitors (Tacrolimus 0.03-0.1%)",
            "Phosphodiesterase-4 inhibitors (Crisaborole)",
            "Antihistamines (Cetirizine, Loratadine)",
            "Moisturizers with ceramides",
            "Wet wrap therapy for severe cases",
            "Dupilumab injections for moderate-severe cases"
        ],
        "precautions": [
            "Identify and avoid triggers (soaps, detergents, allergens)",
            "Short, lukewarm showers (5-10 minutes)",
            "Apply moisturizer immediately after bathing",
            "Use fragrance-free products",
            "Wear soft, breathable fabrics (cotton)",
            "Keep nails short to prevent scratching",
            "Use humidifier in dry climates"
        ]
    },
    "Cellulitis/Impetigo": {
        "medications": [
            "Oral antibiotics (Cephalexin, Dicloxacillin)",
            "Topical mupirocin (for impetigo)",
            "IV antibiotics for severe cases (Vancomycin)",
            "Pain relievers (Acetaminophen, Ibuprofen)",
            "Antiseptic washes (Chlorhexidine)",
            "Warm compresses for comfort"
        ],
        "precautions": [
            "Complete full course of antibiotics",
            "Keep affected area clean and covered",
            "Avoid scratching or touching lesions",
            "Wash hands frequently",
            "Do not share personal items (towels, razors)",
            "Monitor for fever or spreading redness",
            "Treat underlying conditions (eczema, athlete's foot)"
        ]
    },
    "Disorders of Pigmentation": {
        "medications": [
            "Hydroquinone (2-4%)",
            "Topical retinoids (Tretinoin)",
            "Azelaic acid (15-20%)",
            "Chemical peels (glycolic acid)",
            "Laser therapy (Q-switched lasers)",
            "Vitamin C serums",
            "Corticosteroids for inflammatory pigmentation"
        ],
        "precautions": [
            "Daily broad-spectrum sunscreen SPF 30+",
            "Avoid picking at skin lesions",
            "Gradual treatment approach to avoid irritation",
            "Protect skin from trauma/friction",
            "Manage hormonal imbalances if present",
            "Be patient - results take months",
            "Avoid harsh scrubs or exfoliants"
        ]
    },
    "Eczema": {
        "medications": [
            "Topical corticosteroids (Hydrocortisone 1%)",
            "Calcineurin inhibitors (Tacrolimus 0.03-0.1%)",
            "Antihistamines (Cetirizine, Loratadine)",
            "Moisturizers with ceramides",
            "Wet wrap therapy for severe cases",
            "Dupilumab injections for moderate-severe cases",
            "Antibiotics for secondary infections"
        ],
        "precautions": [
            "Avoid triggers (soaps, detergents, allergens)",
            "Short, lukewarm showers",
            "Apply moisturizer immediately after bathing",
            "Use fragrance-free products",
            "Wear soft, breathable fabrics",
            "Use humidifier in dry weather",
            "Manage stress through relaxation techniques"
        ]
    },
    "Exanthems": {
        "medications": [
            "Antihistamines (Diphenhydramine)",
            "Topical corticosteroids (Hydrocortisone 1%)",
            "Oral corticosteroids for severe cases",
            "Antipyretics for fever (Acetaminophen)",
            "Antiviral medications if viral cause",
            "Calamine lotion for itching"
        ],
        "precautions": [
            "Identify and avoid causative agent",
            "Keep skin cool and moisturized",
            "Wear loose, comfortable clothing",
            "Stay hydrated",
            "Monitor for signs of infection",
            "Isolate if contagious condition",
            "Use mild, fragrance-free cleansers"
        ]
    },
    "Fungal Infections": {
        "medications": [
            "Topical antifungals (Clotrimazole, Miconazole)",
            "Oral antifungals (Terbinafine, Fluconazole)",
            "Antifungal shampoos (Ketoconazole 2%)",
            "Nail lacquers for onychomycosis",
            "Powders for prevention in shoes",
            "Oral itraconazole for extensive infections"
        ],
        "precautions": [
            "Keep affected areas clean and dry",
            "Change socks and underwear daily",
            "Avoid walking barefoot in public areas",
            "Disinfect shoes and clothing",
            "Treat all affected areas simultaneously",
            "Complete full course of treatment",
            "Manage underlying conditions (diabetes)"
        ]
    },
    "Herpes/HPV": {
        "medications": [
            "Antivirals (Acyclovir, Valacyclovir)",
            "Topical creams (Docosanol, Penciclovir)",
            "HPV vaccine (Gardasil 9)",
            "Podophyllotoxin for genital warts",
            "Imiquimod for warts",
            "Cryotherapy for warts",
            "Pain relievers (Ibuprofen, Acetaminophen)"
        ],
        "precautions": [
            "Avoid skin-to-skin contact during outbreaks",
            "Practice safe sex with barriers",
            "Don't share personal items (razors, towels)",
            "Manage stress which can trigger outbreaks",
            "Keep affected areas clean and dry",
            "Avoid picking at lesions",
            "Sun protection for oral herpes"
        ]
    },
    "Lupus": {
        "medications": [
            "Hydroxychloroquine (first-line treatment)",
            "Topical corticosteroids",
            "Oral corticosteroids (Prednisone)",
            "Immunosuppressants (Methotrexate, Azathioprine)",
            "Belimumab (Biologic therapy)",
            "NSAIDs for pain and inflammation",
            "Vitamin D supplements"
        ],
        "precautions": [
            "Strict sun protection (SPF 50+, protective clothing)",
            "Regular monitoring by rheumatologist",
            "Balanced diet with anti-inflammatory foods",
            "Gentle skin care with mild products",
            "Manage stress and get adequate rest",
            "Avoid smoking and excessive alcohol",
            "Monitor for signs of disease flares"
        ]
    },
    "Melanoma Skin Cancer": {
        "medications": [
            "Surgical excision (primary treatment)",
            "Immunotherapy (Pembrolizumab, Nivolumab)",
            "Targeted therapy (BRAF inhibitors)",
            "Chemotherapy for advanced cases",
            "Radiation therapy for specific cases",
            "Sentinel lymph node biopsy for staging"
        ],
        "precautions": [
            "Immediate dermatologist referral required",
            "Monthly self-skin exams (ABCDE rule)",
            "Professional skin exams every 3-6 months",
            "Strict sun protection (SPF 50+, clothing)",
            "Avoid tanning beds completely",
            "Monitor for new or changing moles",
            "Genetic counseling if family history"
        ]
    },
    "Poison Ivy": {
        "medications": [
            "Topical corticosteroids (Hydrocortisone 1%)",
            "Oral corticosteroids for severe cases",
            "Oral antihistamines (Diphenhydramine)",
            "Calamine lotion for itching",
            "Colloidal oatmeal baths",
            "Cool compresses",
            "Antibiotics if secondary infection"
        ],
        "precautions": [
            "Wash skin immediately after exposure",
            "Wash all clothing and tools that contacted plant",
            "Avoid scratching to prevent infection",
            "Keep fingernails clean and short",
            "Learn to identify poison ivy/oak/sumac",
            "Wear protective clothing when outdoors",
            "Use barrier creams if high-risk exposure"
        ]
    },
    "Psoriasis": {
        "medications": [
            "Topical corticosteroids",
            "Vitamin D analogs (Calcipotriene)",
            "Topical retinoids (Tazarotene)",
            "Phototherapy (UVB, PUVA)",
            "Systemic medications (Methotrexate, Cyclosporine)",
            "Biologics (Adalimumab, Secukinumab)",
            "Coal tar preparations"
        ],
        "precautions": [
            "Moisturize skin regularly",
            "Avoid triggers (stress, infections, injuries)",
            "Limit alcohol consumption",
            "Don't pick at scales or plaques",
            "Moderate sun exposure may help (with caution)",
            "Maintain healthy weight",
            "Join support groups for chronic management"
        ]
    },
    "Seborrheic Keratoses": {
        "medications": [
            "Cryotherapy (liquid nitrogen)",
            "Electrocautery",
            "Curettage (scraping off)",
            "Laser ablation",
            "Topical hydrogen peroxide (40%)",
            "Topical tazarotene for some cases"
        ],
        "precautions": [
            "No treatment needed unless symptomatic",
            "Avoid picking or scratching lesions",
            "Monitor for changes in appearance",
            "Gentle skin care to avoid irritation",
            "Sun protection may prevent new lesions",
            "See dermatologist if diagnosis uncertain",
            "Document lesions with photos for tracking"
        ]
    },
    "Systemic Disease": {
        "medications": [
            "Depends on underlying condition",
            "Immunosuppressants for autoimmune",
            "Antihistamines for urticarial manifestations",
            "Corticosteroids for inflammation",
            "DMARDs for rheumatoid arthritis",
            "Biologics for specific conditions",
            "Supportive therapies for symptoms"
        ],
        "precautions": [
            "Comprehensive medical evaluation needed",
            "Manage underlying systemic condition",
            "Regular follow-up with specialists",
            "Monitor for new or changing symptoms",
            "Maintain detailed health records",
            "Sun protection if photosensitive",
            "Balanced diet and regular exercise"
        ]
    },
    "Urticaria Hives": {
        "medications": [
            "Second-gen antihistamines (Loratadine, Cetirizine)",
            "First-gen antihistamines at night (Diphenhydramine)",
            "H2 blockers (Famotidine) for refractory cases",
            "Oral corticosteroids for severe episodes",
            "Omalizumab for chronic urticaria",
            "Leukotriene inhibitors (Montelukast)",
            "Topical antipruritics (Menthol, Camphor)"
        ],
        "precautions": [
            "Identify and avoid triggers (foods, medications)",
            "Keep symptom diary to identify patterns",
            "Wear loose, comfortable clothing",
            "Use cool compresses for relief",
            "Avoid hot showers/baths during outbreaks",
            "Manage stress which can exacerbate",
            "Carry emergency epinephrine if history of anaphylaxis"
        ]
    },
    "Vascular Tumors": {
        "medications": [
            "Propranolol for infantile hemangiomas",
            "Timolol gel for superficial lesions",
            "Oral corticosteroids for problematic cases",
            "Laser therapy (Pulsed dye laser)",
            "Surgical excision for specific cases",
            "Sclerotherapy for venous malformations",
            "Sirolimus for complex vascular anomalies"
        ],
        "precautions": [
            "Dermatologist evaluation for proper diagnosis",
            "Monitor for ulceration or bleeding",
            "Protect lesions from trauma",
            "Document changes with photography",
            "Genetic counseling for hereditary forms",
            "Sun protection for fragile skin",
            "Multidisciplinary care for complex cases"
        ]
    },
    "Vasculitis": {
        "medications": [
            "Corticosteroids (Prednisone)",
            "Immunosuppressants (Methotrexate, Azathioprine)",
            "Colchicine for Behçet's disease",
            "IVIG for certain types",
            "Rituximab for refractory cases",
            "NSAIDs for mild cases",
            "Antibiotics if infection-associated"
        ],
        "precautions": [
            "Prompt medical evaluation is essential",
            "Monitor for systemic involvement",
            "Protect skin from trauma/injury",
            "Regular blood pressure monitoring",
            "Vaccinations as recommended (avoid live vaccines if immunosuppressed)",
            "Dental care to prevent oral ulcers",
            "Smoking cessation critical"
        ]
    },
    "Warts Molluscum": {
        "medications": [
            "Salicylic acid preparations",
            "Cryotherapy (liquid nitrogen)",
            "Cantharidin solution",
            "Imiquimod cream (for molluscum)",
            "Podophyllotoxin (for genital warts)",
            "Curettage (scraping off)",
            "Laser therapy for resistant cases"
        ],
        "precautions": [
            "Avoid picking or shaving over lesions",
            "Don't share personal items (towels, razors)",
            "Keep affected areas clean and dry",
            "Wear flip-flops in public showers",
            "Practice safe sex for genital warts",
            "Boost immune system with healthy lifestyle",
            "Be patient - treatment may take months"
        ]
    }
}

        
        self.setup_ui()
        self.style_ui()
        
    def setup_ui(self):
        # Create main container
        self.main_container = tk.Frame(self.root, bg="#f5f5f5")
        self.main_container.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Header
        self.header_frame = tk.Frame(self.main_container, bg="#4a6fa5")
        self.header_frame.pack(fill="x", pady=(0, 20))
        
        self.title_label = tk.Label(
            self.header_frame,
            text="Skin Disease Detection System",
            font=("Segoe UI", 24, "bold"),
            bg="#4a6fa5",
            fg="white",
            padx=20,
            pady=15
        )
        self.title_label.pack()
        
        # Content area
        self.content_frame = tk.Frame(self.main_container, bg="#f5f5f5")
        self.content_frame.pack(expand=True, fill="both")
        
        # Left panel - Image display
        self.image_frame = tk.Frame(self.content_frame, bg="white", bd=2, relief="groove")
        self.image_frame.pack(side="left", expand=True, fill="both", padx=(0, 10), pady=10)
        
        self.image_label = tk.Label(
            self.image_frame,
            text="Input Image",
            font=("Segoe UI", 16, "bold"),
            bg="white",
            pady=10
        )
        self.image_label.pack()
        
        self.image_display = tk.Label(self.image_frame, bg="white")
        self.image_display.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Right panel - Results
        self.results_frame = tk.Frame(self.content_frame, bg="white", bd=2, relief="groove")
        self.results_frame.pack(side="right", expand=True, fill="both", padx=(10, 0), pady=10)
        
        self.results_label = tk.Label(
            self.results_frame,
            text="Detection Results",
            font=("Segoe UI", 16, "bold"),
            bg="white",
            pady=10
        )
        self.results_label.pack()
        
        # Results container with scrollbar
        self.results_canvas = tk.Canvas(self.results_frame, bg="white", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_canvas.yview)
        self.scrollable_frame = tk.Frame(self.results_canvas, bg="white")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            )
        )
        
        self.results_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.results_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Result widgets
        self.disease_label = tk.Label(
            self.scrollable_frame,
            text="",
            font=("Segoe UI", 14, "bold"),
            bg="white",
            fg="#333333",
            pady=10
        )
        self.disease_label.pack(fill="x", padx=20)
        
        self.confidence_label = tk.Label(
            self.scrollable_frame,
            text="",
            font=("Segoe UI", 12),
            bg="white",
            fg="#555555"
        )
        self.confidence_label.pack(fill="x", padx=20, pady=(0, 10))
        
        # Medications section
        self.medications_frame = tk.Frame(self.scrollable_frame, bg="white", bd=1, relief="solid")
        self.medications_frame.pack(fill="x", padx=20, pady=(10, 5))
        
        self.medications_title = tk.Label(
            self.medications_frame,
            text="Recommended Medications:",
            font=("Segoe UI", 12, "bold"),
            bg="#f0f8ff",
            fg="#333333",
            padx=10,
            pady=5,
            anchor="w"
        )
        self.medications_title.pack(fill="x")
        
        self.medications_text = tk.Label(
            self.medications_frame,
            text="",
            font=("Segoe UI", 11),
            bg="white",
            fg="#333333",
            justify="left",
            wraplength=400,
            padx=10,
            pady=10
        )
        self.medications_text.pack(fill="x")
        
        # Precautions section
        self.precautions_frame = tk.Frame(self.scrollable_frame, bg="white", bd=1, relief="solid")
        self.precautions_frame.pack(fill="x", padx=20, pady=(5, 10))
        
        self.precautions_title = tk.Label(
            self.precautions_frame,
            text="Precautions and Care:",
            font=("Segoe UI", 12, "bold"),
            bg="#f0f8ff",
            fg="#333333",
            padx=10,
            pady=5,
            anchor="w"
        )
        self.precautions_title.pack(fill="x")
        
        self.precautions_text = tk.Label(
            self.precautions_frame,
            text="",
            font=("Segoe UI", 11),
            bg="white",
            fg="#333333",
            justify="left",
            wraplength=400,
            padx=10,
            pady=10
        )
        self.precautions_text.pack(fill="x")
        
        # Button panel
        self.button_frame = tk.Frame(self.main_container, bg="#f5f5f5")
        self.button_frame.pack(fill="x", pady=(10, 0))
        
        self.upload_btn = self.create_button(
            self.button_frame,
            "Upload Image",
            self.select_image,
            "#4a6fa5"
        )
        self.upload_btn.pack(side="left", padx=10, pady=10)
        
        self.capture_btn = self.create_button(
            self.button_frame,
            "Capture from Webcam",
            self.capture_from_webcam,
            "#5a8f5a"
        )
        self.capture_btn.pack(side="left", padx=10, pady=10)
        
        self.clear_btn = self.create_button(
            self.button_frame,
            "Clear",
            self.clear_results,
            "#a56f4a"
        )
        self.clear_btn.pack(side="right", padx=10, pady=10)
        
        # Status bar
        self.status_bar = tk.Label(
            self.main_container,
            text="Ready",
            bd=1,
            relief="sunken",
            anchor="w",
            font=("Segoe UI", 10),
            bg="#e0e0e0",
            fg="#333333"
        )
        self.status_bar.pack(fill="x", pady=(10, 0))
        
    def style_ui(self):
        # Configure styles
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure scrollbar style
        style.configure("Vertical.TScrollbar", 
                       background="#d0d0d0", 
                       bordercolor="#d0d0d0",
                       arrowcolor="#333333",
                       troughcolor="#e0e0e0")
        
        # Configure button styles
        style.configure("TButton", 
                       font=("Segoe UI", 11),
                       padding=6,
                       relief="flat")
        
        style.map("TButton",
                background=[("active", "#e0e0e0")],
                foreground=[("active", "black")])
        
    def create_button(self, parent, text, command, color):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="white",
            font=("Segoe UI", 12, "bold"),
            relief="raised",
            padx=15,
            pady=8,
            bd=0,
            cursor="hand2",
            activebackground=color,
            activeforeground="white"
        )
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )
        if file_path:
            try:
                image = Image.open(file_path)
                self.classify_image(image)
                self.status_bar.config(text=f"Loaded image: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_bar.config(text="Error loading image")
    
    def capture_from_webcam(self):
        self.status_bar.config(text="Initializing webcam...")
        self.root.update()
        
        try:
            cap = cv2.VideoCapture(1)
            
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                self.status_bar.config(text="Webcam error")
                return
                
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                self.classify_image(image)
                self.status_bar.config(text="Image captured from webcam")
            else:
                messagebox.showerror("Error", "Failed to capture image")
                self.status_bar.config(text="Capture failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Webcam error: {str(e)}")
            self.status_bar.config(text="Webcam error")
    
    def classify_image(self, image):
        # Resize and preprocess image
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Display the image
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        self.image_display.config(image=photo)
        self.image_display.image = photo
        
        # Prepare for prediction
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Make prediction
        prediction = self.model.predict(data)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]
        
        # Update results
        self.disease_label.config(text=f"Detected Condition: {class_name}")
        self.confidence_label.config(
            text=f"Confidence: {confidence_score * 100:.2f}%",
            fg="#007acc" if confidence_score > 0.7 else "#cc0000"
        )
        
        # Show recommendations
        recommendations = self.MEDICAL_RECOMMENDATIONS.get(class_name, {})
        medications = "\n• ".join(recommendations.get("medications", ["No specific recommendations available"]))
        precautions = "\n• ".join(recommendations.get("precautions", ["No specific precautions available"]))
        
        self.medications_text.config(text=f"• {medications}")
        self.precautions_text.config(text=f"• {precautions}")
        
        # Adjust scroll region
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        self.results_canvas.yview_moveto(0)
        
        self.status_bar.config(text="Analysis complete")
    
    def clear_results(self):
        self.image_display.config(image="", text="No image loaded")
        self.image_display.image = None
        self.disease_label.config(text="")
        self.confidence_label.config(text="")
        self.medications_text.config(text="")
        self.precautions_text.config(text="")
        self.status_bar.config(text="Ready")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = SkinDiseaseApp(root)
    app.run()
