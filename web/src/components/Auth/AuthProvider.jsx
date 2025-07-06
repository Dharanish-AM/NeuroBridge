import React from 'react';
import { AuthContext, useAuthState } from '../../hooks/useAuth';

const AuthProvider = ({ children }) => {
  const authMethods = useAuthState();

  return (
    <AuthContext.Provider value={authMethods}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthProvider;