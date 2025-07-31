import { Redirect } from "react-router-dom";

export default function Index() {
  // Redirect from the root to the dashboard page
  return <Redirect to="/dashboard" />;
}