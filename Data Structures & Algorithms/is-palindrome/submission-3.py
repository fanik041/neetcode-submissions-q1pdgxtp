class Solution:
    def isPalindrome(self, s: str) -> bool:
        
        def is_alNum(c:str) -> bool:
            val = ord(c)
            return (
                48 <= val <= 57 or
                65 <= val <= 90 or
                97 <= val <= 122
            )

        
        def to_lower(c: str) -> str:
            val = ord(c)
            if 65 <= val <= 90:
                return chr(val + 32)
            return c


        left = 0
        right = len(s) - 1

        while left < right:
            while left < right and not is_alNum(s[left]):
                left += 1
            
            while left < right and not is_alNum(s[right]):
                right -=1

            if to_lower(s[left]) != to_lower(s[right]):
                return False
            
            left +=1
            right -=1

        return True