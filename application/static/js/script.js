document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const gpaDisplay = document.getElementById('gpa-value');
    const ctx = document.getElementById('gpa-chart').getContext('2d');
    
    // Initialize chart
    const gpaChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Study Time', 'Attendance', 'Engagement', 'Other Factors'],
            datasets: [{
                data: [25, 25, 25, 25],
                backgroundColor: [
                    '#4361ee',
                    '#4cc9f0',
                    '#4895ef',
                    '#3f37c9'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            cutout: '70%',
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = {
            StudyTimeWeekly: document.getElementById('StudyTimeWeekly').value,
            Absences: document.getElementById('Absences').value,
            Tutoring: document.getElementById('Tutoring').value,
            ParentalSupport: document.getElementById('ParentalSupport').value,
            EngagementIndex: document.getElementById('EngagementIndex').value,
            AttendanceRate: document.getElementById('AttendanceRate').value
        };
        
        try {
            // call and post the form
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            //retrieve the data
            const data = await response.json();
            

            //update the chart
            if (data.status === 'success') {
                gpaDisplay.textContent = data.predicted_gpa;
                
                
                gpaChart.data.datasets[0].data = [
                    Math.abs(formData.StudyTimeWeekly * 0.3),
                    Math.abs(formData.AttendanceRate * 0.4),
                    Math.abs(formData.EngagementIndex * 0.2),
                    Math.abs(formData.ParentalSupport * 0.1)
                ];

                const gradeElement = document.getElementById('grade-value');
                 if (gradeElement) {
                    gradeElement.textContent = data.predicted_grade;
                }
                gpaChart.update();
                
                // Color code GPA result
                const gpa = parseFloat(data.predicted_gpa);
                gpaDisplay.style.color = 
                    gpa >= 3.5 ? '#2ecc71' : 
                    gpa >= 2.0 ? '#f39c12' : '#e74c3c';
            } else {
                alert('Prediction failed: ' + data.message);
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    });
});