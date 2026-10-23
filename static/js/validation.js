/**
 * Client-Side Validation for Churn Prediction Form
 */

document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("churnPredictionForm");
  if (!form) return;

  const totalOrders = document.getElementById("total_orders");
  const recencyDays = document.getElementById("recency_days");
  const avgOrderInterval = document.getElementById("avg_order_interval");
  const avgBasketSize = document.getElementById("avg_basket_size");
  const reorderRatio = document.getElementById("reorder_ratio");
  const totalProducts = document.getElementById("total_products");
  const totalReorders = document.getElementById("total_reorders");

  // Auto-calculate suggested products & reorders if left blank
  function updateCalculatedEstimates() {
    const orders = parseFloat(totalOrders.value) || 0;
    const basket = parseFloat(avgBasketSize.value) || 0;
    const ratio = parseFloat(reorderRatio.value) || 0;

    if (totalProducts && (!totalProducts.value || totalProducts.dataset.autoFilled === "true")) {
      if (orders > 0 && basket > 0) {
        const estProducts = Math.round(orders * basket);
        totalProducts.value = estProducts;
        totalProducts.dataset.autoFilled = "true";
      }
    }

    if (totalReorders && (!totalReorders.value || totalReorders.dataset.autoFilled === "true")) {
      const prodCount = parseFloat(totalProducts ? totalProducts.value : 0) || 0;
      if (prodCount > 0 && ratio >= 0) {
        const estReorders = Math.round(prodCount * ratio);
        totalReorders.value = Math.min(estReorders, prodCount);
        totalReorders.dataset.autoFilled = "true";
      }
    }
  }

  [totalOrders, avgBasketSize, reorderRatio].forEach((input) => {
    if (input) {
      input.addEventListener("input", updateCalculatedEstimates);
    }
  });

  if (totalProducts) {
    totalProducts.addEventListener("input", () => {
      delete totalProducts.dataset.autoFilled;
    });
  }
  if (totalReorders) {
    totalReorders.addEventListener("input", () => {
      delete totalReorders.dataset.autoFilled;
    });
  }

  // Form submission validation
  form.addEventListener("submit", function (event) {
    let isValid = true;
    const errorList = [];

    // Clear previous invalid states
    form.querySelectorAll(".is-invalid").forEach((el) => el.classList.remove("is-invalid"));

    function markInvalid(input, message) {
      if (!input) return;
      input.classList.add("is-invalid");
      const feedback = input.parentElement.querySelector(".invalid-feedback");
      if (feedback) feedback.textContent = message;
      errorList.push(message);
      isValid = false;
    }

    const ordersVal = parseInt(totalOrders.value, 10);
    if (isNaN(ordersVal) || ordersVal < 1 || ordersVal > 150) {
      markInvalid(totalOrders, "Total orders must be between 1 and 150.");
    }

    const recencyVal = parseFloat(recencyDays.value);
    if (isNaN(recencyVal) || recencyVal < 0 || recencyVal > 180) {
      markInvalid(recencyDays, "Recency must be between 0 and 180 days.");
    }

    const intervalVal = parseFloat(avgOrderInterval.value);
    if (isNaN(intervalVal) || intervalVal < 0 || intervalVal > 60) {
      markInvalid(avgOrderInterval, "Average order interval must be between 0 and 60 days.");
    }

    const basketVal = parseFloat(avgBasketSize.value);
    if (isNaN(basketVal) || basketVal < 1 || basketVal > 100) {
      markInvalid(avgBasketSize, "Average basket size must be between 1 and 100 items.");
    }

    const ratioVal = parseFloat(reorderRatio.value);
    if (isNaN(ratioVal) || ratioVal < 0.0 || ratioVal > 1.0) {
      markInvalid(reorderRatio, "Reorder ratio must be between 0.0 and 1.0.");
    }

    const prodVal = parseInt(totalProducts.value, 10);
    const reorderVal = parseInt(totalReorders.value, 10);
    if (!isNaN(prodVal) && !isNaN(reorderVal)) {
      if (reorderVal > prodVal) {
        markInvalid(totalReorders, "Total reorders cannot exceed total products.");
      }
    }

    if (!isValid) {
      event.preventDefault();
      event.stopPropagation();
      const alertContainer = document.getElementById("clientValidationAlert");
      if (alertContainer) {
        alertContainer.classList.remove("d-none");
        alertContainer.innerHTML = `<strong>Please correct the following:</strong><ul class="mb-0 mt-1">${errorList
          .map((e) => `<li>${e}</li>`)
          .join("")}</ul>`;
      }
    }
  });
});
