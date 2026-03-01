import js from "@eslint/js";

export default [
  js.configs.recommended,
  {
    files: ["**/*.js"],
    languageOptions: {
      globals: {
        chrome: "readonly",
        indexedDB: "readonly",
        crypto: "readonly",
        document: "readonly",
        window: "readonly",
        location: "readonly",
        fetch: "readonly",
        URL: "readonly",
        Blob: "readonly",
        FormData: "readonly",
        console: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        btoa: "readonly",
        TextEncoder: "readonly",
        MutationObserver: "readonly",
        Element: "readonly",
        HTMLImageElement: "readonly",
      },
      ecmaVersion: 2022,
      sourceType: "module",
    },
    rules: {
      "no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
      "no-undef": "error"
    }
  }
];
