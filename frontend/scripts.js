// ================= ALL PAGE LOGIC =================

document.addEventListener("DOMContentLoaded", function () {

    // ================= NAVBAR ACTIVE LINK =================

    const currentPage = window.location.pathname.split("/").pop();

    document.querySelectorAll(".nav-link").forEach(link => {
        if (link.getAttribute("href") === currentPage) {
            link.classList.add("text-blue-600", "font-semibold", "active");
        }
    });

    // ================= MOBILE MENU TOGGLE =================

    const menuBtn = document.getElementById("menu-btn");
    const mobileMenu = document.getElementById("mobile-menu");

    if (menuBtn && mobileMenu) {
        menuBtn.addEventListener("click", () => {
            mobileMenu.classList.toggle("hidden");
        });
    }

    // ================= FILE PREVIEW (Detection Page) =================

    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const uploadStatus = document.getElementById("uploadStatus");

    if (fileInput) {
        fileInput.addEventListener("change", function () {

            if (fileInput.files.length === 0) {
                uploadStatus.innerHTML = "⚠️ File not uploaded!!!";
                uploadStatus.className = "mb-4 text-center text-red-600 font-medium";
                previewContainer.classList.add("hidden");
                return;
            }

            const file = fileInput.files[0];

            if (file.type === "application/pdf") {
                uploadStatus.innerHTML = "📄 PDF uploaded successfully!";
                uploadStatus.className = "mb-4 text-center text-green-600 font-medium";
                previewContainer.classList.add("hidden");
            } 
            else if (file.type.startsWith("image/")) {
                uploadStatus.innerHTML = "✅ Image uploaded successfully!";
                uploadStatus.className = "mb-4 text-center text-green-600 font-medium";

                const reader = new FileReader();
                reader.onload = function (e) {
                    imagePreview.src = e.target.result;
                    previewContainer.classList.remove("hidden");
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // ================= EMAILJS CONTACT FORM =================

    const contactForm = document.getElementById("contactForm");

    if (contactForm) {

        // Initialize EmailJS
        emailjs.init("6vI5Vj_yMf4mUG6J-");  // Your Public Key

        const formStatus = document.getElementById("formStatus");

        contactForm.addEventListener("submit", function (e) {
            e.preventDefault();

            formStatus.innerHTML = "Sending...";
            formStatus.className = "text-gray-600 font-medium";

            const templateParams = {
                user_name: document.getElementById("user_name").value,
                user_email: document.getElementById("user_email").value,
                message: document.getElementById("message").value
            };

            emailjs.send(
                "service_ifv7giq",      // Your Service ID
                "template_k7rhu2g",     // Your Template ID
                templateParams
            )
            .then(function () {
                formStatus.innerHTML = "✅ Message sent successfully!";
                formStatus.className = "text-green-600 font-semibold";
                contactForm.reset();
            })
            .catch(function (error) {
                formStatus.innerHTML = "❌ Failed to send message.";
                formStatus.className = "text-red-600 font-semibold";
                console.error("EmailJS Error:", error);
            });

        });
    }

});


// ================= PREDICTION FUNCTION =================

async function predictTumor() {

    const fileInput = document.getElementById("fileInput");
    const resultDiv = document.getElementById("result");

    if (!fileInput || !fileInput.files.length) {
        resultDiv.innerHTML = "⚠️ Please upload a file first!";
        resultDiv.className = "mt-6 text-center text-red-600 font-semibold";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultDiv.innerHTML = "Processing...";
    resultDiv.className = "mt-6 text-center text-gray-500 font-semibold";

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = data.error;
            resultDiv.className = "mt-6 text-center text-red-600 font-semibold";
        }
        else if (data.prediction === "Tumor") {
            resultDiv.innerHTML = "🚨 Tumor Detected";
            resultDiv.className = "mt-6 text-center text-red-600 font-bold text-xl";
        }
        else {
            resultDiv.innerHTML = "✅ No Tumor";
            resultDiv.className = "mt-6 text-center text-green-600 font-bold text-xl";
        }

    } catch (error) {
        resultDiv.innerHTML = "Server error. Please try again.";
        resultDiv.className = "mt-6 text-center text-red-600 font-semibold";
    }
}