function onDelete() {
    let confirmation = document.getElementById("confirmation");
    if (!confirmation.classList.contains("modal-open")) {
      confirmation.classList.add("modal-open");
    }
  }

  function onCancel() {
    let confirmation = document.getElementById("confirmation");
    confirmation.classList.remove("modal-open");
  }

  function onConfirm() {
    onCancel();
  }

  document.addEventListener("DOMContentLoaded", () => {
    document
      .getElementById("confirmation")
      .addEventListener("click", onCancel);
    document
      .querySelector(".modal")
      .addEventListener("click", (e) => e.stopPropagation());
  });