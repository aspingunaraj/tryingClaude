package com.example.demo;

import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Map;

@Service
public class KiteService {

    static final String API_KEY = "9a4fnxekfbrw0k6d";
    private static final String API_SECRET = "d4avfhwjneq1kkxqhfj06rfcyhgykng4";
    private static final String KITE_BASE_URL = "https://api.kite.trade";

    public String getLoginUrl() {
        return "https://kite.zerodha.com/connect/login?api_key=" + API_KEY + "&v=3";
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> generateSession(String requestToken) throws Exception {
        String checksum = sha256(API_KEY + requestToken + API_SECRET);

        RestTemplate restTemplate = new RestTemplate();

        HttpHeaders headers = new HttpHeaders();
        headers.set("X-Kite-Version", "3");
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("api_key", API_KEY);
        body.add("request_token", requestToken);
        body.add("checksum", checksum);

        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(body, headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(
                KITE_BASE_URL + "/session/token", request, Map.class);

        Map<String, Object> responseBody = response.getBody();
        if (responseBody != null && "success".equals(responseBody.get("status"))) {
            return (Map<String, Object>) responseBody.get("data");
        }
        throw new RuntimeException("Failed to generate session: " + responseBody);
    }

    private String sha256(String input) throws NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] hash = md.digest(input.getBytes(StandardCharsets.UTF_8));
        StringBuilder hex = new StringBuilder();
        for (byte b : hash) {
            String h = Integer.toHexString(0xff & b);
            if (h.length() == 1) hex.append('0');
            hex.append(h);
        }
        return hex.toString();
    }
}
