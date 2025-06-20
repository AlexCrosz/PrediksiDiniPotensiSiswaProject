from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler dari file .pkl
with open('model.pkl', 'rb') as f_model:
    model = pickle.load(f_model)

with open('scaler.pkl', 'rb') as f_scaler:
    scaler = pickle.load(f_scaler)

def give_recommendations(grade, student_data):
    recommendations_dict = {
        "A": "Luar biasa! Pertahankan kinerja ini. Teruskan belajar dengan konsisten dan coba eksplorasi topik lebih lanjut.",
        "B": "Bagus, tetapi ada ruang untuk peningkatan. Fokuskan pada area yang masih sedikit sulit dan pertahankan disiplin belajar.",
        "C": "Perlu perhatian lebih. Cobalah untuk lebih fokus pada materi yang belum dikuasai, ikuti bimbingan atau les tambahan.",
        "D": f"Perlu peningkatan yang signifikan. Berikut adalah beberapa rekomendasi berdasarkan data Anda:\n"
              f"- Cobalah untuk meningkatkan waktu belajar. Anda belajar hanya {student_data.get('StudyTimeWeekly', '0')} jam per minggu, coba tambahkan hingga 10-15 jam.\n"
              f"- Perhatikan absensi Anda. Anda telah absen sebanyak {student_data.get('Absences', '0')} kali, usahakan untuk hadir lebih sering.\n"
              f"- Pertimbangkan untuk meningkatkan dukungan orang tua dan mencari tutor atau les tambahan.",
        "E": f"Perlu bantuan segera. Berikut adalah beberapa rekomendasi berdasarkan data Anda:\n"
              f"- Waktu belajar Anda terbilang rendah ({student_data.get('StudyTimeWeekly', '0')} jam per minggu). Cobalah untuk belajar lebih banyak dan lebih terstruktur.\n"
              f"- Absensi Anda tinggi, yaitu {student_data.get('Absences', '0')} kali. Pastikan untuk hadir lebih sering di kelas.\n"
              f"- Dukungan orang tua juga rendah. Cobalah untuk berdiskusi antara Anak dan Orang Tua tentang dukungan yang bisa diberikan."
    }
    return recommendations_dict.get(grade, "Rekomendasi tidak tersedia.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    recommendation = None
    form_data = {}

    if request.method == 'POST':
        try:
            # Ambil data form
            form_data = request.form.to_dict()

            # Daftar fitur yang model harapkan dan urutannya harus sesuai
            feature_order = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly',
                             'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular',
                             'Sports', 'Music', 'Volunteering']

            # Konversi nilai input ke float dan buat array fitur sesuai urutan
            features = []
            for feat in feature_order:
                val = form_data.get(feat)
                if val is None or val == '':
                    raise ValueError(f'Field {feat} tidak boleh kosong.')
                features.append(float(val))

            # Skala fitur menggunakan scaler.pkl
            features_scaled = scaler.transform([features])

            # Prediksi kelas menggunakan model.pkl
            pred_class_index = model.predict(features_scaled)[0]

            # Mapping prediksi model (asumsi model mengeluarkan angka 0-4 yang sesuai dengan huruf A-E)
            mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
            prediction = mapping.get(pred_class_index, 'Unknown')

            # Dapatkan rekomendasi berupa string
            recommendation = give_recommendations(prediction, form_data)

        except Exception as e:
            prediction = f"Terjadi kesalahan: {str(e)}"
            recommendation = None

    return render_template('index.html', prediction=prediction, recommendation=recommendation, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
