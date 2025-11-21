// Елементи сторінок
const page1 = document.getElementById('page-1');
const page2 = document.getElementById('page-2');

// Елементи керування
const actionButton = document.getElementById('action-btn');
const userInput = document.getElementById('user-input');
const backButton = document.getElementById('back-btn');

// Елементи другої сторінки
const resultText = document.getElementById('result-text');
const backButton2 = document.getElementById('back-btn-2');
const newComplaintInput = document.getElementById('new-complaint-input');
const newComplaintBtn = document.getElementById('new-complaint-btn');

// Перехід з 1 сторінки на 2
actionButton.addEventListener('click', function() {
    const text = userInput.value;

    if (text.trim() === "") {
        alert("Введіть текст заяви!");
        return;
    }

    resultText.innerText = text;
    
    page1.style.display = 'none';
    page2.style.display = 'flex'; 
});

// Стрілки "Назад"
backButton.addEventListener('click', function() {
    console.log("Назад з головної");
});

backButton2.addEventListener('click', function() {
    page2.style.display = 'none';
    page1.style.display = 'flex';
});

// 3. Логіка кнопки "Надіслати ще раз" на другій сторінці
newComplaintBtn.addEventListener('click', function() {
    const newText = newComplaintInput.value;

    // Перевірка
    if (newText.trim() === "") {
        alert("Напишіть нову скаргу в нижньому полі!");
        return;
    }

    // Оновлення тексту
    resultText.innerText = newText;

    // Очищення поля вводу
    newComplaintInput.value = "";
});