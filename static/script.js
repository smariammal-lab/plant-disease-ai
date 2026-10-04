const form = document.getElementById("predictionForm");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");

// Image Preview
imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    const imageURL = URL.createObjectURL(file);

    preview.innerHTML = `
        <p><strong>Selected Image:</strong></p>
        <img src="${imageURL}" alt="Tomato Leaf">
    `;

    result.innerHTML = "";
});


// Prediction
form.addEventListener("submit", async function (event) {

    event.preventDefault();

    const file = imageInput.files[0];

    if (!file) {
        result.innerHTML = `
            <div class="error-box">
                ⚠️ Please select a tomato leaf image.
            </div>
        `;
        return;
    }

    result.innerHTML = `
        <div class="loading-box">
            🔄 Analyzing image...<br>
            Please wait.
        </div>
    `;

    const formData = new FormData();
    formData.append("image", file);

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {

            result.innerHTML = `
                <div class="error-box">
                    ❌ ${data.error}
                </div>
            `;

            return;
        }

        const confidence = Number(data.confidence);

        let confidenceMessage = "";

        if (confidence < 60) {

            confidenceMessage = `
                <div class="warning">
                    ⚠️ Low confidence prediction.<br>
                    Please upload a clear leaf image or verify
                    the result with an agricultural expert.
                </div>
            `;

        } else if (confidence < 85) {

            confidenceMessage = `
                <div class="warning">
                    ⚠️ Moderate confidence prediction.
                </div>
            `;

        } else {

            confidenceMessage = `
                <div class="success">
                    ✅ High confidence prediction.
                </div>
            `;
        }


        result.innerHTML = `
            <div class="result-card">

                <h2>🌱 Prediction Result</h2>

                <h3>
                    ${data.disease_name}
                </h3>

                <p>
                    📊 <strong>Confidence:</strong>
                    ${confidence}%
                </p>

                ${confidenceMessage}

                <div class="advisory">

                    <h4>💡 Advisory</h4>

                    <p>
                        ${data.advisory}
                    </p>

                </div>

            </div>
        `;

    } catch (error) {

        console.error(error);

        result.innerHTML = `
            <div class="error-box">
                ❌ Unable to connect to the prediction server.
                Please make sure Flask is running.
            </div>
        `;
    }

});
const resetBtn = document.getElementById("resetBtn");

resetBtn.addEventListener("click", function () {

    imageInput.value = "";

    preview.innerHTML = "";

    result.innerHTML = "";


    
});