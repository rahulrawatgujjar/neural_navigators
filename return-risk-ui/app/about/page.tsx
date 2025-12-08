export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-800 via-slate-600 to-slate-800 flex items-center justify-center p-6">
      <div className="max-w-4xl bg-white rounded-3xl shadow-2xl p-10">
        
        <h1 className="text-3xl font-bold text-center text-slate-800 mb-6">
          About This Project
        </h1>

        <p className="text-slate-600 text-lg leading-relaxed mb-6 text-center">
          The <b>Return Risk Prediction System</b> is a machine learning-based
          project developed during the <b>HCL Hackathon at NIT Kurukshetra
          (NIT KKR)</b>. The goal of this project is to predict whether a customer
          is likely to return a purchased product, helping businesses reduce
          return rates and improve overall profitability.
        </p>

        {/* Hackathon Info */}
        <div className="bg-slate-100 rounded-xl p-6 mb-6">
          <h2 className="text-xl font-semibold mb-2 text-slate-800">
            🏆 Hackathon Details
          </h2>
          <ul className="list-disc list-inside space-y-1 text-slate-700 text-sm">
            <li><b>Event:</b> HCL Hackathon</li>
            <li><b>Institute:</b> National Institute of Technology, Kurukshetra (NIT KKR)</li>
            <li><b>Team Name:</b> Neural Navigators</li>
          </ul>
        </div>

        {/* Technologies & Features */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-slate-700 mb-6">
          
          <div className="bg-slate-100 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-2">🔹 Technologies Used</h2>
            <ul className="list-disc list-inside space-y-1 text-sm">
              <li>Next.js (Frontend)</li>
              <li>FastAPI (Backend)</li>
              <li>Machine Learning (Scikit-learn)</li>
              <li>Tailwind CSS</li>
              <li>Python</li>
            </ul>
          </div>

          <div className="bg-slate-100 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-2">🔹 Key Features</h2>
            <ul className="list-disc list-inside space-y-1 text-sm">
              <li>Real-time return prediction</li>
              <li>Probability-based confidence score</li>
              <li>Low / High risk classification</li>
              <li>Interactive UI</li>
              <li>ML model comparison</li>
            </ul>
          </div>

        </div>

        {/* Team Section */}
        <div className="bg-slate-100 rounded-xl p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 text-slate-800 text-center">
            👨‍💻 Team Neural Navigators
          </h2>

          <ul className="space-y-2 text-center text-slate-700 font-medium">
            <li>Rahul Rawat</li>
            <li>Sourabh Vishwakarma</li>
            <li>Kumar Utkarsh</li>
          </ul>
        </div>

        {/* Closing Line */}
        <div className="text-center text-slate-600">
          <p>
            This project demonstrates the application of machine learning to
            solve real-world business problems in the field of e-commerce
            analytics.
          </p>
        </div>

      </div>
    </div>
  );
}
