"use client";
import { useState } from "react";

type FormData = {
  Product_Category: string;
  Product_Price: string;
  Order_Quantity: string;
  Discount_Applied: string;
  User_Age: string;
  User_Gender: string;
  User_Location: string;
  Payment_Method: string;
  Shipping_Method: string;
};

export default function PredictionForm() {
  const [formData, setFormData] = useState<FormData>({
    Product_Category: "",
    Product_Price: "",
    Order_Quantity: "",
    Discount_Applied: "",
    User_Age: "",
    User_Gender: "",
    User_Location: "",
    Payment_Method: "",
    Shipping_Method: ""
  });

  const [result, setResult] = useState<number | null>(null);
  const [probability, setProbability] = useState<number | null>(null); // ✅ NEW
  const [loading, setLoading] = useState(false);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setProbability(null); // ✅ RESET PROBABILITY

    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData)
    });

    const data = await res.json();

    setResult(data.prediction);
    setProbability(data.probability); // ✅ STORE PROBABILITY
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-6">
      <div className="w-full max-w-3xl bg-white/95 backdrop-blur rounded-3xl shadow-2xl p-10">
        <h1 className="text-3xl font-bold text-center text-slate-800 mb-2">
          Return Risk Prediction
        </h1>
        <p className="text-center text-slate-500 mb-8">
          Predict whether a customer is likely to return a product
        </p>

        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-5">
          
          {/* Product Category */}
          <div>
            <label className="form-label">Product Category</label>
            <input name="Product_Category" onChange={handleChange} className="form-input" required />
          </div>

          {/* Product Price */}
          <div>
            <label className="form-label">Product Price</label>
            <input type="number" name="Product_Price" onChange={handleChange} className="form-input" required />
          </div>

          {/* Order Quantity */}
          <div>
            <label className="form-label">Order Quantity</label>
            <input type="number" name="Order_Quantity" onChange={handleChange} className="form-input" required />
          </div>

          {/* Discount */}
          <div>
            <label className="form-label">Discount Applied</label>
            <input type="number" name="Discount_Applied" onChange={handleChange} className="form-input" required />
          </div>

          {/* User Age */}
          <div>
            <label className="form-label">User Age</label>
            <input type="number" name="User_Age" onChange={handleChange} className="form-input" required />
          </div>

          {/* Gender */}
          <div>
            <label className="form-label">User Gender</label>
            <select name="User_Gender" onChange={handleChange} className="form-input" required>
              <option value="">Select</option>
              <option>Male</option>
              <option>Female</option>
            </select>
          </div>

          {/* Location */}
          <div>
            <label className="form-label">User Location</label>
            <input name="User_Location" onChange={handleChange} className="form-input" required />
          </div>

          {/* Payment */}
          <div>
            <label className="form-label">Payment Method</label>
            <select name="Payment_Method" onChange={handleChange} className="form-input" required>
              <option value="">Select</option>
              <option>Credit Card</option>
              <option>Debit Card</option>
              <option>PayPal</option>
              <option>Gift Card</option>
              <option>COD</option>
            </select>
          </div>

          {/* Shipping */}
          <div className="md:col-span-2">
            <label className="form-label">Shipping Method</label>
            <select name="Shipping_Method" onChange={handleChange} className="form-input" required>
              <option value="">Select</option>
              <option>Standard</option>
              <option>Express</option>
              <option>Next-Day</option>
            </select>
          </div>

          {/* Submit Button */}
          <div className="md:col-span-2 mt-6">
            <button
              type="submit"
              className="w-full bg-gradient-to-r from-indigo-600 to-blue-600 text-white font-semibold py-3 rounded-xl hover:scale-[1.02] active:scale-[0.98] transition-transform shadow-lg"
            >
              {loading ? "Predicting..." : "Predict Return Risk"}
            </button>
          </div>
        </form>

        {/* ✅ RESULT + ✅ PROBABILITY DISPLAY */}
        {result !== null && probability !== null && (
          <div
            className={`mt-8 text-center text-lg font-semibold px-6 py-4 rounded-xl ${
              result === 1
                ? "bg-red-100 text-red-700 border border-red-300"
                : "bg-green-100 text-green-700 border border-green-300"
            }`}
          >
            <p>
              {result === 1
                ? "⚠️ High Risk: Product Likely to be Returned"
                : "✅ Low Risk: Product Likely to be Kept"}
            </p>
                
            <p className="mt-2 text-base font-medium">
              Return Probability: <b>{(probability * 100).toFixed(2)}%</b>
            </p>

            {/* ✅ Simple Probability Bar */}
            <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
              <div
                className="bg-indigo-600 h-2 rounded-full"
                style={{ width: `${probability * 100}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Tailwind re-usable classes */}
      <style jsx>{`
        .form-label {
          display: block;
          margin-bottom: 0.25rem;
          font-size: 0.875rem;
          font-weight: 600;
          color: #334155;
        }
        .form-input {
          width: 100%;
          padding: 0.6rem 0.75rem;
          border-radius: 0.75rem;
          border: 1px solid #cbd5f5;
          outline: none;
          transition: all 0.2s;
        }
        .form-input:focus {
          ring: 2px;
          border-color: #6366f1;
          box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }
      `}</style>
    </div>
  );
}
