// ============================================
// TTT CRM - Frontend Activity Tracker
// ADD THIS TO: frontend/index.html (bottom of <body>)
// OR include as: <script src="/static/crm-tracker.js"></script>
// ============================================

(function () {
  const API = "";  // empty = same domain (your Railway backend)
  const SESSION_KEY = "ttt_session_id";

  // ── Generate/get session ID ──────────────────────────
  function getSessionId() {
    let s = sessionStorage.getItem(SESSION_KEY);
    if (!s) {
      s = "sess_" + Math.random().toString(36).slice(2) + Date.now();
      sessionStorage.setItem(SESSION_KEY, s);
    }
    return s;
  }

  // ── Get current user ID (email if logged in, anon otherwise) ──
  function getUserId() {
    // Try localStorage for logged-in user email
    const user = localStorage.getItem("ttt_user");
    if (user) {
      try {
        const parsed = JSON.parse(user);
        return parsed.email || ("anon_" + getSessionId());
      } catch (e) {}
    }
    return "anon_" + getSessionId();
  }

  // ── Core: send activity to backend ──────────────────
  function track(activityType, data = {}) {
    const payload = {
      user_id:       getUserId(),
      session_id:    getSessionId(),
      activity_type: activityType,
      activity_data: data,
      page_url:      window.location.pathname,
    };
    // Fire and forget — non-blocking
    fetch(API + "/api/crm/activity", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      keepalive: true,
    }).catch(() => {}); // silent fail — never break the UX
  }

  // ── Track page visits ────────────────────────────────
  // Fires on every page load, with 30-second debounce per page
  (function trackVisit() {
    const pageKey = "ttt_visited_" + window.location.pathname;
    const lastVisit = parseInt(sessionStorage.getItem(pageKey) || "0");
    const now = Date.now();
    if (now - lastVisit > 30000) {  // 30 seconds debounce
      sessionStorage.setItem(pageKey, now.toString());
      track("visit", {
        page:     window.location.pathname,
        referrer: document.referrer,
        title:    document.title,
      });
    }
  })();

  // ── Expose global TTTTracker object ─────────────────
  // Your existing JS code calls these functions on events

  window.TTTTracker = {

    // Call on successful login
    onLogin: function (userData) {
      // Sync profile to CRM
      fetch(API + "/api/crm/sync-profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email:     userData.email,
          name:      userData.name || userData.full_name,
          phone:     userData.phone,
          linkedin:  userData.linkedin,
          instagram: userData.instagram,
          source:    userData.source || document.referrer || "direct",
        }),
        keepalive: true,
      }).catch(() => {});

      track("login", {
        email:  userData.email,
        method: userData.method || "email",
      });
    },

    // Call on signup
    onSignup: function (userData) {
      fetch(API + "/api/crm/sync-profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email:    userData.email,
          name:     userData.name,
          phone:    userData.phone,
          source:   userData.source || new URLSearchParams(window.location.search).get("utm_source") || "organic",
        }),
        keepalive: true,
      }).catch(() => {});

      track("signup", { email: userData.email });
    },

    // Call every time user sends a message to the AI
    onAIChat: function (query, destination) {
      track("chat", {
        query:       query.slice(0, 200),  // truncate for storage
        destination: destination || "",
      });
    },

    // Call when user creates or saves an itinerary
    onTripPlanned: function (destination, days, budget) {
      track("trip", {
        destination: destination,
        days:        days,
        budget:      budget,
      });
    },

    // Call when user searches flights/hotels
    onSearch: function (searchType, query) {
      track("search", {
        type:  searchType,  // 'flights' | 'hotels' | 'destinations'
        query: query,
      });
    },

    // Call when a booking is completed
    onBooking: function (bookingDetails) {
      track("booking", {
        type:        bookingDetails.type,        // 'flight' | 'hotel' | 'package'
        destination: bookingDetails.destination,
        amount:      bookingDetails.amount,
        booking_id:  bookingDetails.id,
      });
    },

    // Call when user views a listing
    onListingView: function (listingName, listingType) {
      track("visit", {
        page:     "listing",
        listing:  listingName,
        type:     listingType,
      });
    },
  };

  console.log("[TTT CRM Tracker] Active ✓ User:", getUserId());

})();


// ============================================
// HOW TO USE IN YOUR EXISTING FRONTEND CODE:
// ============================================
//
// On Login:
//   TTTTracker.onLogin({ email: user.email, name: user.name });
//
// On Signup:
//   TTTTracker.onSignup({ email: user.email, name: user.name, phone: user.phone });
//
// When AI chat is sent:
//   TTTTracker.onAIChat(messageText, "Goa");
//
// When trip is saved:
//   TTTTracker.onTripPlanned("Goa", 3, 30000);
//
// When search happens:
//   TTTTracker.onSearch("flights", "Mumbai to Goa");
//
// When booking completes:
//   TTTTracker.onBooking({ type: "hotel", destination: "Goa", amount: 12000, id: "BK001" });
//
// ============================================
