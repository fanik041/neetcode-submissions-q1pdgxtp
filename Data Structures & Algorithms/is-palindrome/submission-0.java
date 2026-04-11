class Solution {
    public boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;

        while (left < right) {
            while (left < right && !isAlnum(s.charAt(left))) {
                left++;
            }

            while (left < right && !isAlnum(s.charAt(right))) {
                right--;
            }

            if (toLower(s.charAt(left)) != toLower(s.charAt(right))) {
                return false;
            }

            left++;
            right--;
        }

        return true;
    }

    private boolean isAlnum(char c) {
        int val = (int) c;
        return (val >= 48 && val <= 57) ||
               (val >= 65 && val <= 90) ||
               (val >= 97 && val <= 122);
    }

    private char toLower(char c) {
        int val = (int) c;
        if (val >= 65 && val <= 90) {
            return (char) (val + 32);
        }
        return c;
    }
}