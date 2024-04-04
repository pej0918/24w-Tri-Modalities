class YMCA:
    def __init__(self, device) -> None:
        self.device = device
        self.device_msg = 'Tested on GPU.' if 'cuda' in self.device else 'Tested on CPU.'
    
    def get_css(self):
        return """
        img {
            margin: 0 auto;
            display:block;
        }
        h1 {
            text-align: center;
            display:block;
        }
        h3 {
            text-align: center;
            display:block;
        }
        """