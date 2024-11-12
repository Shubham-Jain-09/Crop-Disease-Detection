document.addEventListener("DOMContentLoaded", () => {
    const hamburgerBtn = document.querySelector(".hamburger-btn");
    const closeBtn = document.querySelector(".close-btn");
    const links = document.querySelector(".links");
    const blurOverlay = document.querySelector(".blur-bg-overlay");
    const formPopup = document.querySelector(".form-popup");
    const loginLink = document.querySelector("#login-link");
    const signupLink = document.querySelector("#signup-link");

    console.log({
        hamburgerBtn,
        closeBtn,
        links,
        blurOverlay,
        formPopup,
        loginLink,
        signupLink
    });

    if (hamburgerBtn && closeBtn && links && blurOverlay && formPopup && loginLink && signupLink) {
        hamburgerBtn.addEventListener("click", () => {
            links.classList.toggle("active");
            blurOverlay.classList.toggle("active");
        });

        closeBtn.addEventListener("click", () => {
            links.classList.remove("active");
            blurOverlay.classList.remove("active");
            formPopup.classList.remove("show-popup");
        });

        loginLink.addEventListener("click", (event) => {
            event.preventDefault();
            formPopup.classList.add("show-popup");
            blurOverlay.classList.add("active");
            formPopup.classList.remove("show-signup");
            formPopup.classList.add("show-login");
            window.location.href = '/login';
        });

        signupLink.addEventListener("click", (event) => {
            event.preventDefault();
            formPopup.classList.add("show-popup");
            blurOverlay.classList.add("active");
            formPopup.classList.remove("show-login");
            formPopup.classList.add("show-signup");
            window.location.href = '/signup';
        });

        document.addEventListener("click", (event) => {
            if (!formPopup.contains(event.target) && !event.target.matches('.hamburger-btn') && !event.target.matches('.close-btn')) {
                formPopup.classList.remove("show-popup");
                blurOverlay.classList.remove("active");
            }
        });
    } else {
        console.warn("One or more elements not found in the DOM");
    }
});
