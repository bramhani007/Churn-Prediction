/**
 * Dashboard Chart.js Visualizations and Metric Updates
 */

document.addEventListener("DOMContentLoaded", function () {
  const riskCtx = document.getElementById("riskDoughnutChart");
  const probCtx = document.getElementById("probBarChart");

  let riskChart = null;
  let probChart = null;

  function initCharts(metrics) {
    if (!metrics) return;

    // 1. Risk Distribution Doughnut Chart
    if (riskCtx) {
      const riskData = metrics.risk_distribution || { Low: 0, Medium: 0, High: 0 };
      const dataValues = [riskData.Low || 0, riskData.Medium || 0, riskData.High || 0];

      if (riskChart) riskChart.destroy();

      riskChart = new Chart(riskCtx, {
        type: "doughnut",
        data: {
          labels: ["Low Risk (<35%)", "Medium Risk (35-65%)", "High Risk (>65%)"],
          datasets: [
            {
              data: dataValues,
              backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
              hoverBackgroundColor: ["#059669", "#d97706", "#dc2626"],
              borderWidth: 2,
              borderColor: "#ffffff"
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "bottom",
              labels: {
                boxWidth: 12,
                font: { size: 12, weight: "500" },
                padding: 16
              }
            },
            tooltip: {
              callbacks: {
                label: function (context) {
                  const total = context.dataset.data.reduce((a, b) => a + b, 0);
                  const val = context.raw || 0;
                  const pct = total > 0 ? ((val / total) * 100).toFixed(1) : 0;
                  return ` ${context.label}: ${val} (${pct}%)`;
                }
              }
            }
          },
          cutout: "68%"
        }
      });
    }

    // 2. Churn Probability Histogram Bar Chart
    if (probCtx) {
      const histData = metrics.probability_histogram || [0, 0, 0, 0, 0];

      if (probChart) probChart.destroy();

      probChart = new Chart(probCtx, {
        type: "bar",
        data: {
          labels: ["0 - 20%", "20 - 40%", "40 - 60%", "60 - 80%", "80 - 100%"],
          datasets: [
            {
              label: "Customers",
              data: histData,
              backgroundColor: [
                "rgba(16, 185, 129, 0.7)",
                "rgba(59, 130, 246, 0.7)",
                "rgba(245, 158, 11, 0.7)",
                "rgba(249, 115, 22, 0.7)",
                "rgba(239, 68, 68, 0.8)"
              ],
              borderColor: ["#10b981", "#3b82f6", "#f59e0b", "#f97316", "#ef4444"],
              borderWidth: 1,
              borderRadius: 6
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: function (ctx) {
                  return ` Customers in range: ${ctx.raw}`;
                }
              }
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              ticks: { precision: 0 },
              grid: { color: "#f1f5f9" }
            },
            x: {
              grid: { display: false }
            }
          }
        }
      });
    }
  }

  // Check if initial metrics are supplied on page in a JSON script element
  const initialDataElem = document.getElementById("initialMetricsData");
  if (initialDataElem) {
    try {
      const metrics = JSON.parse(initialDataElem.textContent);
      initCharts(metrics);
    } catch (e) {
      console.warn("Could not parse inline metrics; fetching from API:", e);
      fetchMetrics();
    }
  } else {
    fetchMetrics();
  }

  function fetchMetrics() {
    fetch("/api/metrics")
      .then((res) => res.json())
      .then((json) => {
        if (json.status === "success" && json.data) {
          initCharts(json.data);
        }
      })
      .catch((err) => console.error("Error fetching live metrics:", err));
  }
});
