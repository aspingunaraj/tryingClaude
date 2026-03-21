package com.example.demo;

import jakarta.servlet.http.HttpSession;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.util.Map;
import java.text.NumberFormat;
import java.util.Locale;

@Controller
public class KiteController {

    private final KiteService kiteService;

    public KiteController(KiteService kiteService) {
        this.kiteService = kiteService;
    }

    @GetMapping("/")
    public String index(HttpSession session, Model model) {
        String accessToken = (String) session.getAttribute("accessToken");
        if (accessToken != null) {
            model.addAttribute("loggedIn", true);
            model.addAttribute("userName", session.getAttribute("userName"));
            model.addAttribute("userEmail", session.getAttribute("userEmail"));
            model.addAttribute("userId", session.getAttribute("userId"));
            model.addAttribute("broker", session.getAttribute("broker"));
            model.addAttribute("loginTime", session.getAttribute("loginTime"));
            model.addAttribute("accessToken", maskToken(accessToken));

            // Fetch live fund balance
            try {
                Map<String, Object> funds = kiteService.getFunds(accessToken);
                @SuppressWarnings("unchecked")
                Map<String, Object> equity = (Map<String, Object>) funds.get("equity");
                if (equity != null) {
                    model.addAttribute("equityNet", formatCurrency(equity.get("net")));
                    @SuppressWarnings("unchecked")
                    Map<String, Object> available = (Map<String, Object>) equity.get("available");
                    if (available != null) {
                        model.addAttribute("equityCash", formatCurrency(available.get("cash")));
                    }
                }
            } catch (Exception ignored) {
                // funds not critical — dashboard still loads
            }
        } else {
            model.addAttribute("loggedIn", false);
            model.addAttribute("loginUrl", kiteService.getLoginUrl());
        }
        return "index";
    }

    @GetMapping("/redirect")
    public String handleRedirect(
            @RequestParam(required = false) String request_token,
            @RequestParam(required = false, defaultValue = "success") String status,
            HttpSession session,
            RedirectAttributes redirectAttributes) {

        if (!"success".equals(status) || request_token == null) {
            redirectAttributes.addFlashAttribute("error", "Login was cancelled or failed. Please try again.");
            return "redirect:/";
        }

        try {
            Map<String, Object> data = kiteService.generateSession(request_token);
            session.setAttribute("accessToken", (String) data.get("access_token"));
            session.setAttribute("userName", data.get("user_name"));
            session.setAttribute("userEmail", data.get("email"));
            session.setAttribute("userId", data.get("user_id"));
            session.setAttribute("broker", data.get("broker"));
            session.setAttribute("loginTime", data.get("login_time"));
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Authentication failed: " + e.getMessage());
        }

        return "redirect:/";
    }

    @GetMapping("/logout")
    public String logout(HttpSession session) {
        session.invalidate();
        return "redirect:/";
    }

    private String formatCurrency(Object value) {
        if (value == null) return "N/A";
        try {
            double amount = Double.parseDouble(value.toString());
            return String.format("₹%,.2f", amount);
        } catch (NumberFormatException e) {
            return value.toString();
        }
    }

    private String maskToken(String token) {
        if (token == null || token.length() < 8) return "****";
        return token.substring(0, 4) + "..." + token.substring(token.length() - 4);
    }
}
